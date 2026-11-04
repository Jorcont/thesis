''' Basic evaluation metrics for general meta-learning.'''
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import os

import logging
logger = logging.getLogger("meta-netze")

class BasicEvaluations:
    ''' This class receives a dictionary of task-specific
    training and testing losses and evaluates the corresponding performance based
    on the metrics configured within this class.'''
    def __init__(self, config):
        self.calc_generalization = config['generalization']
        self.calc_adaptation_speed = config['adaptation_speed']
        self.calc_sample_efficiency = config['sample_efficiency']
        self.print_results = config['print_results']
        self.gen_errors = []
        self.adaptation_speeds = []

    def __call__(self, losses):
        return self.evaluate(losses)

    def evaluate(self, all_metrics):
        ''' Evaluates performance for the given objects of all_metrics.
        Each entry in all_metrics is assumed to consists of a task_id and the corresponding metrics.
        The metrics dict contains different metrics, always including the losses, for that particular task.
        '''
        self.test_losses = []
        for task in all_metrics:
            if not task:
                continue
            metrics = task[-1]
            if "losses" in metrics and metrics["losses"]["test"]:
                self.test_losses.append(metrics['losses']['test'][-1])
            if self.calc_generalization:
                self.gen_errors.append(self.calculate_generalization_errors(metrics))

        if not self.test_losses:  # ← guard against all tasks being empty
            print("[BasicEvaluations] WARNING: no valid tasks to evaluate")
            return

        self.avg_test_loss = self.average_loss(self.test_losses)
        self.avg_gen_error = self.average_loss(self.gen_errors) if self.gen_errors else 0.0

        if self.print_results:
            self.print_performance_measures()

        # ── Visualization (only if heatmap-compatible metrics exist) ──
        # This requires all tasks to have the same metric keys and step counts.
        # Guard: only attempt if tasks have multi-step data and seaborn is available.
        try:
            self._try_visualize(all_metrics)
        except Exception as e:
            logger.warning(f"[BasicEvaluations] Visualization skipped: {e}")

        # self.visualize_results_cache(all_metrics, [f"task_{i}" for i, task in enumerate(all_metrics)], "test", "all")
        # evaluations = [self.distribution(all_metrics, "average"), self.distribution(all_metrics, "min"), self.distribution(all_metrics, "max"), self.distribution(all_metrics, "variance")]
        # labels = ["average", "min", "max", "variance"]
        # self.visualize_results_cache(evaluations,labels, "test", name = "distribution", steps =3)
        # self.visualize_results_cache(evaluations, labels, "train", name="distribution")

    def _try_visualize(self, all_metrics):
        """Attempt heatmap visualization. Safe to fail for regression tasks."""
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Get output path from config
        from conf.configs import PipelineParams
        import os
        config = PipelineParams.get_config()
        output_dir = str(config.output_folder) + "/metric_outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Check if we have enough structured data for heatmaps
        valid_tasks = [t for t in all_metrics if t and len(t) > 0]
        if len(valid_tasks) < 2:
            return

        # Get available metric keys from first valid task
        first_metrics = valid_tasks[0][-1]
        available_keys = list(first_metrics.keys())

        labels = [f"task_{i}" for i in range(len(valid_tasks))]

        for key in available_keys:
            for mode in ["train", "test"]:
                # Check if this key/mode combination has data
                has_data = all(
                    key in t[-1] and mode in t[-1][key] and len(t[-1][key][mode]) > 0
                    for t in valid_tasks
                )
                if not has_data:
                    continue

                # Build data matrix
                max_len = max(len(t[-1][key][mode]) for t in valid_tasks)
                data = np.array([
                    t[-1][key][mode] + [np.nan] * (max_len - len(t[-1][key][mode]))
                    for t in valid_tasks
                ])

                # Plot
                n_tasks = len(valid_tasks)
                if n_tasks > 20:
                    continue  # skip huge heatmaps

                fig, ax = plt.subplots(figsize=(max(6, max_len * 0.8), max(4, n_tasks * 0.4)))
                sns.heatmap(data, annot=True, fmt=".3f", cmap="Blues",
                            ax=ax, linewidths=0.5)
                ax.set_title(f"{key} ({mode})")
                ax.set_xlabel("adaptation step")
                ax.set_ylabel("task")
                plt.tight_layout()
                fig.savefig(f"{output_dir}/{mode}_{key}_heatmap.png", dpi=100)
                plt.close()


    # def visualize_results_cache(self, all_metrics: list,labels: list, mode: str, name:str, steps: int =1):
    #     print("befor_visualize:", all_metrics[0][5]["accuracy"][mode])
    #     for key in all_metrics[0][0]:
    #         print(name, key)
    #         if len(all_metrics) > 5:
    #             fig, axes = plt.subplots(math.ceil(len(all_metrics) / 5), 5, figsize=(30, 18))
    #             axes = axes.flatten()
    #         else:
    #             fig, axes = plt.subplots(1,len(all_metrics), figsize = (30,18), squeeze = False)
    #             axes = axes.flatten()
    #         for task_number, task in enumerate(all_metrics):
    #             max_len = max(len(task[i][key][mode][::steps]) for i in range(len(task)))
    #             data = np.array([task[i][key][mode][::steps]+[np.nan]*(max_len-len(task[i][key][mode])) for i in range(len(task))])
    #             if key == "accuracy" and task_number == 0:
    #                 print(task[5][key][mode])
    #                 print(data)
    #             sns.heatmap(data, annot = True, fmt = ".2f", cmap = "Blues", ax = axes[task_number], square = True, linewidths=0.5)
    #             axes[task_number].set_title(labels[task_number])
    #             axes[task_number].set_xlabel("adaptation_step")
    #             axes[task_number].set_ylabel("sample_step")
    #         fig.savefig(f"/mnt/data/kes/data/ldorsch/meta-netze/metric_outputs/{name}_{mode}_{key}_heatmap.png")

        #if self.print_results:
         #   self.print_performance_measures()

    def average_loss(self, losses):
        ''' Average the given list of task-speciic loss
        to yield the overall average loss.'''
        return sum(losses)/len(losses)

    def distribution(self, all_metrics, dist):
        '''Computes avarage, max, min or variance of metrics.

        Attributes:
             dist: chooses which kind of computation is made. Chooses from max, min, avarage, variance.
             all_metrics: list of metrics over all batches.
        '''
        results = copy.deepcopy(all_metrics[0])
        for task in range(len(all_metrics[0])):
            for key in all_metrics[0][task].keys():
                for mode in ["train", "test"]:
                    for i in range(len(all_metrics[0][task][key][mode])):
                        if dist == "average":
                            results[task][key][mode][i] =sum(all_metrics[j][task][key][mode][i] for j in range(len(all_metrics)))/len(all_metrics)
                        elif dist == "min":
                            results[task][key][mode][i] = min(all_metrics[j][task][key][mode][i] for j in range(len(all_metrics)))
                        elif dist == "max":
                            results[task][key][mode][i] = max(all_metrics[j][task][key][mode][i] for j in range(len(all_metrics)))
                        elif dist == "variance":
                            results[task][key][mode][i] = sum((all_metrics[k][task][key][mode][i]-sum(all_metrics[j][task][key][mode][i] for j in range(len(all_metrics)))/len(all_metrics))**2 for k in range(len(all_metrics)))/len(all_metrics)
                        else:
                            print("wrong metric")
                            return None
        return results

    def calculate_generalization_errors(self, metrics):
        ''' Uses the train and test losses stored in metrics to calculate performance measures.
        '''
        last_train_loss = metrics['losses']['train'][-1] if metrics['losses']['train'] else 0.0
        test_loss = metrics['losses']['test'][-1] if metrics['losses']['test'] else 0.0
        return test_loss - last_train_loss

    def calculate_adaptation_speed(self, all_metrics):
        results = copy.deepcopy(all_metrics)
        for batch in range(len(all_metrics)):
            for task in range(len(all_metrics[0])):
                for key in all_metrics[0][task].keys():
                    for mode in ["train", "test"]:
                        results[batch][task][key][mode]= results[batch][task][key][mode][:-1]
                        for i in range(len(all_metrics[batch][task][key][mode]))[:-1]:
                            results[batch][task][key][mode] =all_metrics[batch][task][key][mode][i]- all_metrics[batch][task][key][mode][i+1]
        return results


    def calculate_sample_efficiency(self, all_metrics):
        ''' Calculates the sample efficiency with the given metrics.'''
        results = all_metrics
        for batch in range(len(all_metrics)):
            for task in range(len(all_metrics[0]))[:-1]:
                results[batch][task] = results[batch][task][:-1]
                for key in all_metrics[0][task].keys():
                    for mode in ["train", "test"]:
                        for i in range(len(all_metrics[batch][task][key][mode]))[:-1]:
                            results[batch][task][key][mode] =all_metrics[batch][task][key][mode][i]- all_metrics[batch][task+1][key][mode][i]
        return results

    def print_performance_measures(self):
        ''' Prints and plots all calculated performance measures in the desired form.
        '''
        print(f"\n[BasicEvaluations] Results:")
        print(f"  Avg test loss       : {self.avg_test_loss:.6f}")
        print(f"  Num test tasks      : {len(self.test_losses)}")
        if self.gen_errors:
            print(f"  Avg gen. gap        : {self.avg_gen_error:.6f}")
        if self.test_losses:
            print(f"  Min test loss       : {min(self.test_losses):.6f}")
            print(f"  Max test loss       : {max(self.test_losses):.6f}")
            print(f"  Std test loss       : {np.std(self.test_losses):.6f}")
