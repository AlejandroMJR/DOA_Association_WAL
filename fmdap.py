from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD, value

def solve_fractional_mdap(simDict, shape, tol=1e-9, max_iters=10):
    """
    Solve the Fractional MDAP problem using the Dinkelbach’s algorithm.

    Args:
    - simDict: dictionary of similarities for every feasible association
    - shape: Array with the number of detections per node {P_n}
    - tol: Tolerance for convergence of the Dinkelbach's algorithm
    - max_iters: Maximum number of iterations for the Dinkelbach's algorithm

    Returns:
    - assigned_features: List of associations (assigment)
    """
    # Create a function to build and solve the LP for a given lambda
    def solve_lp(lambda_val):
        nNodes = len(shape)

        prob = LpProblem("MDAP_Fractional", LpMaximize)
        # Decision variables
        x = {key: LpVariable(f"x_{key}", cat="Binary") for key in simDict.keys()}

        # Assignment constraints, each real feature is assigned exactly once
        for axis in range(nNodes):
            for i in range(shape[axis] - 1):  # for non-dummy indices
                prob += lpSum(x[key] for key in x if key[axis] == i) == 1

        # Objective: maximize sum(sim * x) - lambda * sum(x)
        prob += lpSum(simDict[key] * x[key] for key in simDict.keys()) - lambda_val * lpSum(
            x[key] for key in simDict.keys())
        prob.solve(PULP_CBC_CMD(msg=False))

        # Gather the solution and objective values
        x_sol = {k: x[k].varValue for k in x}
        numerator = sum(simDict[key] * x_sol[key] for key in simDict)
        denominator = sum(x_sol[k] for k in simDict)
        return numerator, denominator, value(prob.objective), x_sol

    # Initialize lambda
    lambda_val = 0.0
    for it in range(max_iters):
        num, den, F_val, x_sol = solve_lp(lambda_val)
        if den == 0:
            print("No assignments were made.")
            return None
        # New candidate value for the objective (average similarity)
        new_lambda = num / den
        # Check convergence: if the improvement is very small, stop.
        if abs(F_val) < tol or abs(new_lambda - lambda_val) < tol:
            print(f"Converged after {it + 1} iterations.")
            assigned_measurements = [key for key in x_sol if x_sol[key] == 1]
            return assigned_measurements  # x_sol gives the assignment, new_lambda is the optimal average similarity
        lambda_val = new_lambda
    print("Max iterations reached without convergence.")
    assigned_measurements = [key for key in x_sol if x_sol[key] == 1]
    return assigned_measurements





