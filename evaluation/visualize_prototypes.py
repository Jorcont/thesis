import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from torch.utils.data import DataLoader
import torch
import heapq


class PrototypeVisualization:
    def __init__(self,image_tensor,adaptation_step, model, label_list = None, predicted_label_list= None):
        self.features = model.features(image_tensor)
        self.head = model.head
        self.proto_tensor = self.head.prototype_activations(self.features)[0]
        self.proto_tensor = self.proto_tensor.clone().detach().cpu().numpy()
        self.image_tensor = image_tensor.cpu().numpy()
        self.label_list = label_list
        self.predicted_label_list = predicted_label_list
        self.adaptation_step = adaptation_step
        self.results = []
        self.logits = self.head.forward(self.features)[0]
        print("self.logits shape:", len(self.logits))
        self.logits = F.softmax(self.logits, dim=-1)
        self.datapath = "/mnt/data/kes/data/ldorsch/meta-netze-new/metric_outputs"
        if predicted_label_list is None:
            self.predicted_label_list = range(image_tensor.shape[0])
        if label_list is None:
            self.label_list = range(image_tensor.shape[0])
        print("labels:", self.label_list, self.predicted_label_list, self.adaptation_step)


    def plot_prototype_overlay_with_sim_legend(self, image_rgb, proto_act, proto_num, img_num, adaptation_step, alpha=0.45):
        image_rgb = np.transpose(image_rgb, (1, 2, 0))
        # image_rgb: H x W x 3 uint8
        # proto_act: h x w raw similarity/activation map from model

        if image_rgb.dtype == np.float32 or image_rgb.dtype == np.float64:
            # Falls Werte im Bereich [0, 1] liegen
            if image_rgb.max() <= 1.0:
                image_rgb = (image_rgb * 255).astype(np.uint8)
            else:
                # Falls Werte bereits im Bereich [0, 255] liegen
                image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

        H, W = image_rgb.shape[:2]
        proto_act = np.asarray(proto_act, dtype=np.float32)

        # Same resize logic you use: cubic, fallback to linear for tiny maps
        interp = cv2.INTER_CUBIC if min(proto_act.shape) >= 4 else cv2.INTER_LINEAR
        hm_raw = cv2.resize(proto_act, (W, H), interpolation=interp).astype(np.float32)

        # Normalize only for cv2 colormap conversion
        vmin, vmax = float(hm_raw.min()), float(hm_raw.max())
        if vmax > vmin:
            hm_norm = np.round((hm_raw - vmin) / (vmax - vmin), 2).astype(np.float32)
        else:
            hm_norm = np.zeros_like(hm_raw, dtype=np.float32)

        # Build RGB heatmap for overlay
        hm_u8 = (hm_norm * 255.0).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(hm_u8, cv2.COLORMAP_VIRIDIS)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

        # Overlay
        overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, heat_rgb, alpha, 0.0)


        self.results.append((image_rgb, hm_raw, overlay, proto_num, img_num))

        return hm_raw, hm_norm, heat_rgb, overlay

    def plot(self, rows, columns, adaptation_step, label):
        if rows*columns != len(self.results):
            print("len", len(self.results))
            raise ValueError("result length doesnt match the number of rows and columns")

        fig, ax = plt.subplots(rows, columns, figsize=(24, 20), squeeze= False)
        print("len", len(self.results))
        for i in range(rows):
            for j in range(columns):
                hm_raw = self.results[i * columns + j][1]
                vmin, vmax = float(hm_raw.min()), float(hm_raw.max())
                im = ax[i][j].imshow(hm_raw, cmap="viridis", vmin=vmin, vmax=vmax)
                ax[i][j].axis("off")
                cbar = fig.colorbar(im, ax=ax[i,j], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(
                    f"{self.datapath}/prototype_images/{label}_proto_adaptation:{adaptation_step}.png")
        plt.close()

        self.results = []

    def plot_all_proto(self, image_rgb_number, proto_tensor, image_tensor, adaptation_step):
        for i in range(proto_tensor.shape[1]):
            self.plot_prototype_overlay_with_sim_legend(image_tensor[image_rgb_number], proto_tensor[image_rgb_number, i],i,image_rgb_number, adaptation_step)

    def plot_all_images(self, proto_number, proto_tensor, image_tensor, adaptation_step):
        for i in range(image_tensor.shape[0]):
            self.plot_prototype_overlay_with_sim_legend(image_tensor[i], proto_tensor[i, proto_number], proto_number,i, adaptation_step)

    def find_highest_similarity(self, image_rgb_number, proto_tensor, image_tensor, adaptation_step):
        similarity_list = []
        for i in range(proto_tensor.shape[1]):
            similarity_list.append(self.plot_prototype_overlay_with_sim_legend(image_tensor[image_rgb_number], proto_tensor[image_rgb_number, i],i,image_rgb_number, adaptation_step, False)[0])
        max_sim = 0
        proto_ind = -1
        for ind, hm_raw in enumerate(similarity_list):
            if hm_raw.max() > max_sim:
                max_sim = hm_raw.max()
                proto_ind = ind
        print(f"highest similarity for image {image_rgb_number}: Proto:{proto_ind}. Similarity: {max_sim}")
        self.plot_prototype_overlay_with_sim_legend(image_tensor[image_rgb_number], proto_tensor[image_rgb_number, proto_ind], proto_ind, image_rgb_number, adaptation_step)

    def find_high_similarity(self, proto_tensor, image_tensor, adaptation_step):
        for i in range(proto_tensor.shape[1]):
            for j in range(image_tensor.shape[0]):
                self.plot_prototype_overlay_with_sim_legend(image_tensor[j], proto_tensor[j, i], i,j, adaptation_step)

    def visualize_prototypes_for_image(self, rows, columns, label: str = "all_protos_for_image_"):
        for i in range(self.image_tensor.shape[0]):
            for j in range(self.proto_tensor.shape[1]):
                    self.plot_prototype_overlay_with_sim_legend(self.image_tensor[i], self.proto_tensor[i, j], j, i, self.adaptation_step)
            self.plot(rows, columns, self.adaptation_step, f"{label}{i}")

    def visualize_images_for_prototype(self, rows, columns, proto_number = -1, label = "all_protos_for_image_"):
        if proto_number == -1:
            indices = range(self.proto_tensor.shape[1])
        else:
            indices = [proto_number]
        for j in indices:
            for i in range(self.image_tensor.shape[0]):
                self.plot_prototype_overlay_with_sim_legend(self.image_tensor[i], self.proto_tensor[i, j], j, i, self.adaptation_step)
            self.plot(rows, columns, self.adaptation_step, f"{label}{j}")


    def visualize_images(self, label: str):
        images = self.image_tensor
        fig, axes = plt.subplots(math.ceil(images.shape[0] / 5), 5, figsize=(15, 9))
        axes = axes.flatten()
        for i in range(images.shape[0]):
            img = images[i].transpose(1, 2, 0)
            img = img.clip(0, 1)
            axes[i].imshow(img)
            axes[i].axis('off')
            print("ligits_i",self.logits[i])
            logits = self.logits[i].tolist()
            logits = [round(x,2) for x in logits]
            print("i =", i)
            print(self.predicted_label_list.shape)
            print("pred_list_i", self.predicted_label_list[i])
            print("label_list_i", self.label_list[i])
            print("axes shape:", axes.shape)
            print("label_list:", self.label_list.shape, self.label_list)
            print("pred_list:", self.predicted_label_list.shape, self.predicted_label_list)
            print("images", images.shape[0])
            axes[i].set_title(f"act:{self.label_list[i]}_pred:{self.predicted_label_list[i]}\nlogits:{logits}")
        fig.savefig(f"{self.datapath}/{label}.png")
        plt.close(fig)

    def compare_prototypes(self):
        prototypes = self.head.prototype_vectors
        print("shape", prototypes.shape)
        prototypes_np = prototypes.detach().cpu().numpy()
        prototypes_np = prototypes_np.reshape(prototypes_np.shape[0], -1)
        similarity_matrix = cosine_similarity(prototypes_np)
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=False, cmap='viridis', vmin=-1, vmax=1)
        plt.title("Kosinus-Ähnlichkeit der Prototypen")
        plt.savefig(f"{self.datapath}/Kosinus_Prototypen{self.adaptation_step}.png")


def collect_highest_images(dataset, model,top_k = 10, batch_size = 32):
    device = model.device
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last= False)

    # ein Min-Heap pro Prototyp: speichert (score, unique_id, image)
    top_per_proto = None
    counter = 0  # tie-breaker, damit Tupel-Vergleich nie am Bild scheitert

    with torch.no_grad():
        for label in dataset.unique_labels:
            X, y = dataset.full_task(label)
            for i in range(0, X.size(0), batch_size):
                images = X[i:i+batch_size]
                images = images.to(device)
                features = model.features(images)
                activations = model.head.prototype_activations(features)[0]
                # activations shape z.B. (batch, num_prototypes, H*W)

                # max-Aktivierung je Prototyp pro Bild
                max_act = activations.amax(dim=(-2,-1))  # (batch, num_prototypes)

                num_prototypes = max_act.shape[1]
                if top_per_proto is None:
                    top_per_proto = [[] for _ in range(num_prototypes)]

                for b in range(images.shape[0]):
                    img_cpu = images[b].clone()  # auf CPU behalten zum Speichern
                    for p in range(num_prototypes):
                        score = max_act[b, p].item()
                        heap = top_per_proto[p]

                        if len(heap) < top_k:
                            heapq.heappush(heap, (score, counter, img_cpu))
                        elif score > heap[0][0]:  # besser als schwächstes im Heap
                            heapq.heapreplace(heap, (score, counter, img_cpu))
                        counter += 1

    # sortiere jeden Heap absteigend nach Score für saubere Ausgabe
    result = []
    for heap in top_per_proto:
        sorted_imgs = sorted(heap, key=lambda x: x[0], reverse=True)
        result.append([(img, score) for score, _, img in sorted_imgs])

    return result  # result[p] = Liste der Top-k (image, score) für Prototyp p


def plot_last_layer_weights(
    ppnet,
    class_names=None,
    figsize=(16, 8),
    cmap="RdBu_r",
    annotate=False,
    sort_prototypes=False,
    save_path=None,
):
    """
    Plot the ProtoPNet last-layer weights as a classes x prototypes heatmap.
    Assumes a SHARED prototype pool (every prototype is used by every class).

    Args:
        ppnet:            The trained ProtoPNet model.
        class_names:      Optional list of class name strings (len == num_classes).
        figsize:          Figure size.
        cmap:             Diverging colormap (RdBu_r centers 0 nicely).
        annotate:         If True, writes weight values in each cell
                          (only feasible for small matrices).
        sort_prototypes:  If True, reorders prototype columns by their total
                          absolute importance (most influential first).
        save_path:        If given, saves the figure to this path.
    """
    # --- 1. Extract the weight matrix: [num_classes, num_prototypes] ---
    W = ppnet.last_layer.weight.detach().cpu().numpy()
    print("W", W)
    num_classes, num_prototypes = W.shape

    # --- 2. Optionally sort prototypes by overall importance ---
    if sort_prototypes:
        importance = np.abs(W).sum(axis=0)          # total influence per prototype
        order = np.argsort(importance)[::-1]        # most important first
        W = W[:, order]
    else:
        order = np.arange(num_prototypes)

    # --- 3. Symmetric color scale centered at 0 ---
    vmax = np.abs(W).max()
    vmin = -vmax

    # --- 4. Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(W, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Weight value", rotation=270, labelpad=15)

    # Axis labels
    ax.set_xlabel(
        "Prototype index" + (" (sorted by importance)" if sort_prototypes else "")
    )
    ax.set_ylabel("Class")
    ax.set_title("ProtoPNet Last-Layer Weights (classes × prototypes)")

    # Y ticks = class names
    ax.set_yticks(np.arange(num_classes))
    if class_names is not None:
        ax.set_yticklabels(class_names)
    else:
        ax.set_yticklabels(np.arange(num_classes))

    # X ticks: show (possibly reordered) prototype indices, thinned if many
    step = max(1, num_prototypes // 40)
    ax.set_xticks(np.arange(0, num_prototypes, step))
    ax.set_xticklabels(order[::step], rotation=90, fontsize=7)

    # --- 5. Optional cell annotations (small matrices only) ---
    if annotate:
        for i in range(num_classes):
            for j in range(num_prototypes):
                ax.text(
                    j, i, f"{W[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=6,
                    color="black" if abs(W[i, j]) < 0.5 * vmax else "white",
                )

    plt.tight_layout()
    if save_path:
        plt.savefig("/mnt/data/kes/data/ldorsch/meta-netze-new/metric_outputs/"+save_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax



