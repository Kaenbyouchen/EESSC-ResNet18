import matplotlib, matplotlib.pyplot as plt
print("OK matplotlib", matplotlib.__version__, "backend:", matplotlib.get_backend())
plt.plot([0,1,2],[0,1,0])
plt.title("matplotlib smoke test")
plt.savefig("matplotlib_test.png", dpi=120)
print("Saved -> matplotlib_test.png")