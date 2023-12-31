import pulp
import numpy as np

def OptimalTransport(C, T):
    problem = pulp.LpProblem("matching", pulp.LpMinimize)
    n, m = T.shape
    Transport = [[pulp.LpVariable("T_"+str(i)+"_"+str(j), 0, 1, "Continuous") for j in range(m)] for i in range(n)]
    for i in range(n):
        problem += sum(Transport[i]) == 1
    for j in range(m):
        problem += sum([row[j] for row in Transport]) == 1
    C_dot_T = [[s*t for (s, t) in zip(C[i], Transport[i])] for i in range(n)]
    total_cost = sum(sum(row) for row in C_dot_T)
    problem += total_cost
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    T_star = np.array([[Transport[i][j].value() for j in range(m)] for i in range(n)])
    misalignment, number_match = 0, 0
    for i in range(n):
        for j in range(m):
            if abs(T_star[i, j] - 1) < 1e-6:
                number_match += 1
                misalignment += C[i][j]
    return T_star, misalignment, number_match