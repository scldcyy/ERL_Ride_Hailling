import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm
import matplotlib.pyplot as plt

from shared_ppo import RideHailingEnv, SharedPPOAgent, PassengerSimulator, CONFIG, DataSetProcesser, Trainer


# --- Strategy Encoder ---
class StrategyEncoder:
    def __init__(self):
        # Gene: [Commission (0.1-0.35), Surge (1.0-3.0), Subsidy (0.0-5.0)]
        self.bounds = np.array([(0.1, 0.35), (1.0, 3.0), (0.0, 5.0)])
        self.dim = 3

    def decode(self, gene):
        # Simplified: Apply same surge/subsidy across all zones/times for demo
        commission, surge, subsidy = gene
        return {
            'commission': commission,
            'lambda': np.full((CONFIG['TIME_STEPS_PER_DAY'], CONFIG['N_ZONES']), surge),
            'subsidy': np.full((CONFIG['TIME_STEPS_PER_DAY'], CONFIG['N_ZONES']), subsidy)
        }


# --- Expensive Evaluation ---
def evaluate_strategy_real(gene,trainer:Trainer):
    encoder = StrategyEncoder()
    platform_params = encoder.decode(gene)

    # Run Simulation
    sim_days=20
    # trainer=Trainer(ppo_weights)
    trainer.train(platform_params,sim_days)
    env = trainer.env
    agent = trainer.agent

    # Calculate Real Objectives
    # Obj 1: Max Profit
    profit = env.platform_profit / sim_days
    # Obj 2: Max accept_rate
    accept_rate = len(env.completed_orders_stats)/len(env.pending_orders)
    # Obj 3: Min Wait Time (Maximize Negative)
    avg_wait = np.mean(env.completed_orders_stats) if env.completed_orders_stats else 30.0

    # Return Objectives (Max, Max, Max) and Weights
    return np.array([profit, accept_rate, -avg_wait]), agent.policy.state_dict()


# --- NSGA-II---
def fast_non_dominated_sort(objectives):
    pop_size = objectives.shape[0]
    S = [[] for _ in range(pop_size)]
    n = np.zeros(pop_size)
    rank = np.zeros(pop_size)
    fronts = [[]]

    for p in range(pop_size):
        for q in range(pop_size):
            # Check if p dominates q (Maximization)
            if np.all(objectives[p] >= objectives[q]) and np.any(objectives[p] > objectives[q]):
                S[p].append(q)
            elif np.all(objectives[q] >= objectives[p]) and np.any(objectives[q] > objectives[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)
    return fronts[:-1], rank


def calculate_crowding_distance(objectives, fronts):
    distances = np.zeros(objectives.shape[0])
    for front in fronts:
        if len(front) == 0: continue
        l = len(front)
        for m in range(objectives.shape[1]):
            # Sort by objective m
            sorted_indices = sorted(front, key=lambda x: objectives[x, m])
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            f_min = objectives[sorted_indices[0], m]
            f_max = objectives[sorted_indices[-1], m]
            if f_max == f_min: continue

            for i in range(1, l - 1):
                distances[sorted_indices[i]] += (objectives[sorted_indices[i + 1], m] - objectives[
                    sorted_indices[i - 1], m]) / (f_max - f_min)
    return distances


def tournament_selection(pop_indices, ranks, distances):
    a, b = np.random.choice(pop_indices, 2, replace=False)
    if ranks[a] < ranks[b]:
        return a
    elif ranks[b] < ranks[a]:
        return b
    else:
        return a if distances[a] > distances[b] else b


def sbx_crossover(p1, p2, bounds, eta=15):
    # Simulated Binary Crossover
    u = np.random.rand()
    if u <= 0.5:
        beta = (2 * u) ** (1.0 / (eta + 1))
    else:
        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))

    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

    c1 = np.clip(c1, bounds[:, 0], bounds[:, 1])
    c2 = np.clip(c2, bounds[:, 0], bounds[:, 1])
    return c1, c2


def polynomial_mutation(p, bounds, eta=20, prob=0.1):
    if np.random.rand() > prob: return p
    mutant = np.copy(p)
    for i in range(len(p)):
        u = np.random.rand()
        if u < 0.5:
            delta = (2 * u) ** (1 / (eta + 1)) - 1
            mutant[i] = p[i] + delta * (p[i] - bounds[i, 0])
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            mutant[i] = p[i] + delta * (bounds[i, 1] - p[i])
    return np.clip(mutant, bounds[:, 0], bounds[:, 1])


# --- Main Surrogate-Assisted Loop ---
def run_main():
    # 1. Setup
    data_proc = DataSetProcesser()

    # Scale drivers/trips for simulation
    scaling = (CONFIG['N_DRIVERS'] * CONFIG['TRIPS_PER_DRIVER_DAY']) / (len(data_proc.df_trips) / 31)  # Approx monthly
    pass_sim = PassengerSimulator(data_proc.df_trips, CONFIG['N_ZONES'], scaling)

    trainer=Trainer()
    trainer.agent.load_by_path('model/agent.pth')

    encoder = StrategyEncoder()
    POP_SIZE = 50
    MAX_GENS = 50

    # 2. Initial Sampling (LHS)
    print("--- Initial Sampling ---")
    sampler = LatinHypercube(d=encoder.dim)
    pop_genes = sampler.random(n=POP_SIZE) * (encoder.bounds[:, 1] - encoder.bounds[:, 0]) + encoder.bounds[:, 0]

    archive_X = []
    archive_Y = []
    archive_W = {}

    # Real Evaluation of Initial Pop
    init_w=trainer.agent.get_weights()
    for i in tqdm(range(POP_SIZE), desc="Eval Initial Pop"):
        fit, w = evaluate_strategy_real(pop_genes[i], trainer)
        archive_X.append(pop_genes[i])
        archive_Y.append(fit)
        archive_W[tuple(pop_genes[i])] = w
        trainer.agent.load_by_weights(init_w)

    # 3. Main Loop
    for gen in tqdm(range(MAX_GENS)):
        print(f"\n--- Generation {gen + 1}/{MAX_GENS} ---")

        # Train Surrogates (One per objective)
        surrogates = [GaussianProcessRegressor(kernel=C(1.0) * RBF(1.0), n_restarts_optimizer=2) for _ in range(3)]
        X_train = np.array(archive_X)
        Y_train = np.array(archive_Y)

        for k in range(3):
            surrogates[k].fit(X_train, Y_train[:, k])

        # --- Virtual Evolution (Using Surrogates) ---
        # Instead of random sampling, we evolve a population using the surrogate
        # to find the best candidate for real evaluation.
        virtual_pop = np.copy(pop_genes)

        for v_gen in range(10):  # 10 generations of virtual evolution
            # Predict fitness
            v_fitness = np.zeros((POP_SIZE, 3))
            for k in range(3):
                v_fitness[:, k] = surrogates[k].predict(virtual_pop)

            # NSGA-II Steps on Virtual Pop
            fronts, ranks = fast_non_dominated_sort(v_fitness)
            dists = calculate_crowding_distance(v_fitness, fronts)

            offspring = []
            while len(offspring) < POP_SIZE:
                p1 = tournament_selection(range(POP_SIZE), ranks, dists)
                p2 = tournament_selection(range(POP_SIZE), ranks, dists)
                c1, c2 = sbx_crossover(virtual_pop[p1], virtual_pop[p2], encoder.bounds)
                offspring.extend([polynomial_mutation(c1, encoder.bounds), polynomial_mutation(c2, encoder.bounds)])
            virtual_pop = np.array(offspring[:POP_SIZE])

        # --- Infill Strategy & Transfer Learning ---
        # Pick the most "unique" or best individual from virtual pop
        # For simplicity: pick random from top front
        v_fitness = np.zeros((POP_SIZE, 3))
        for k in range(3):
            v_fitness[:, k] = surrogates[k].predict(virtual_pop)
        fronts, _ = fast_non_dominated_sort(v_fitness)
        best_candidate_idx = np.random.choice(fronts[0])
        candidate_gene = virtual_pop[best_candidate_idx]

        # Transfer Learning: Find nearest neighbor in archive
        dists = np.linalg.norm(np.array(archive_X) - candidate_gene, axis=1)
        nearest_idx = np.argmin(dists)
        transfer_w = archive_W[tuple(archive_X[nearest_idx])]

        print("Running Expensive Eval with Transfer Learning...")
        trainer.agent.load_by_weights((transfer_w,transfer_w))
        real_fit, real_w = evaluate_strategy_real(candidate_gene, trainer)

        # Update Archive
        archive_X.append(candidate_gene)
        archive_Y.append(real_fit)
        archive_W[tuple(candidate_gene)] = real_w

        # Replace worst in main population with new candidate (Elitism)
        pop_genes[np.random.randint(0, POP_SIZE)] = candidate_gene

    # 4. Results
    print("--- Finished ---")
    front_Y = np.array(archive_Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(front_Y[:, 0], front_Y[:, 1], -front_Y[:, 2])  # Negate wait time for visualization
    ax.set_xlabel('Profit')
    ax.set_ylabel('Order Number')
    ax.set_zlabel('Wait Time')
    plt.savefig('img/3d_plot.png')


if __name__ == '__main__':
    run_main()
