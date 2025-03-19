import gurobipy as gp

# Create a new model
model = gp.Model("test")

# Add a variable
x = model.addVar(name="x")

# Set objective
model.setObjective(x, gp.GRB.MAXIMIZE)

# Add constraint
model.addConstr(x <= 5)

# Optimize model
model.optimize()

# Print solution
if model.status == gp.GRB.OPTIMAL:
    print(f"Optimal solution: x = {x.x}")
else:
    print("No solution found")
