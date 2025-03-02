import pandas as pd
import matplotlib.pyplot as plt

# ✅ Load the saved iteration costs
fixed_costs = pd.read_csv("FIXED_iteration_costs.csv")
dynamic_costs = pd.read_csv("DYNAMIC_iteration_costs.csv")

# ✅ Ensure time column is in datetime format for better plotting
fixed_costs["time"] = pd.to_datetime(fixed_costs["time"])
dynamic_costs["time"] = pd.to_datetime(dynamic_costs["time"])

# ✅ Plot iteration costs comparison
plt.figure(figsize=(14, 6))

plt.plot(fixed_costs["time"], fixed_costs["iteration_cost_value"], label="Fixed Strategy Cost", color="blue", linestyle="-")
plt.plot(dynamic_costs["time"], dynamic_costs["iteration_cost_value"], label="Dynamic Strategy Cost", color="orange", linestyle="--")

plt.xlabel("Time")
plt.ylabel("Iteration Cost (£)")
plt.title("Comparison of Iteration Costs: FIXED vs DYNAMIC Strategies")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

# ✅ Save the plot
plt.tight_layout()
plt.savefig("iteration_costs_comparison.png")
plt.show()

print("✅ Plot saved as 'iteration_costs_comparison.png'")
