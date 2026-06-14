import re
import matplotlib.pyplot as plt
import os

def main():
    log_file_path = os.path.join("MusiqueTestGPTmini", "log.txt")
    
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at {log_file_path}")
        print("Make sure you are running this script from the 'AriGraph' directory.")
        return

    f1_list = []
    recall_list = []
    precision_list = []
    em_list = []
    steps = []

    # Regex to match the metrics line
    # Example: F1: 0.3453093812375249, RECALL: 0.29930795847750863, PRECISION: 0.4080188679245283, EXACT MATCH: 0.235
    metrics_regex = re.compile(
        r"F1:\s*([0-9.]+),\s*RECALL:\s*([0-9.]+),\s*PRECISION:\s*([0-9.]+),\s*EXACT MATCH:\s*([0-9.]+)"
    )

    with open(log_file_path, "r", encoding="utf-8") as f:
        step_counter = 1
        for line in f:
            match = metrics_regex.search(line)
            if match:
                f1_list.append(float(match.group(1)))
                recall_list.append(float(match.group(2)))
                precision_list.append(float(match.group(3)))
                em_list.append(float(match.group(4)))
                steps.append(step_counter)
                step_counter += 1

    if not steps:
        print("No metrics found in the log file.")
        return

    print(f"Extracted metrics for {len(steps)} steps.")

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plotting each metric
    plt.plot(steps, f1_list, label="F1 Score", color="#1f77b4", linewidth=2)
    plt.plot(steps, recall_list, label="Recall", color="#ff7f0e", linewidth=1.5, linestyle="--")
    plt.plot(steps, precision_list, label="Precision", color="#2ca02c", linewidth=1.5, linestyle="-.")
    plt.plot(steps, em_list, label="Exact Match (EM)", color="#d62728", linewidth=2)

    # Title and Labels
    plt.title("AriGraph V3 Reproduction Metrics Evolution (Musique Dataset)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Evaluated Samples (Steps)", fontsize=12)
    plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
    
    # Grid and Legend
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11, loc="best")
    plt.ylim(-0.05, 1.05)
    
    # Save the plot
    output_image_name = "metrics_evolution.png"
    plt.savefig(output_image_name, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Success! Graph saved as '{output_image_name}' in the current directory.")

if __name__ == "__main__":
    main()
