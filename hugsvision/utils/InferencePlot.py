import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime


class InferencePlot:
    """
    🤗 Constructor for the object detection trainer
    """
    def __init__(self, id2label=None, IMG_OUT="./out_img/",
                 show_tag=True, show_confidence=True, show_tags=None,
                 show_plot=False):
        self.id2label = id2label
        self.IMG_OUT = IMG_OUT
        self.show_tag = show_tag
        self.show_confidence = show_confidence
        self.show_tags = show_tags
        self.show_plot = show_plot
        # Colors for visualization
        self.COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125], [0.494, 0.184, 0.556],
                       [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
        if show_tags is None:
            self.show_tags = [ y for x, y in self.id2label.items()]

    def plot_results(self, pil_img, prob, boxes):
        # plt.figure(figsize=(10,16))
        my_dpi = 96
        img_w, img_h = pil_img.size
        plt.figure(figsize=(img_w / my_dpi, img_h / my_dpi), dpi=my_dpi)

        plt.imshow(pil_img)
        ax = plt.gca()

        print(self.id2label)

        colors = self.COLORS * 100

        # print(torchvision.ops.nms(
        #     boxes.cuda(),
        #     prob.cuda(),
        #     0.8,
        # ))

        # For each bbox
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            # Get the highest probability
            cl = p.argmax()
            tag = self.id2label[cl.item()]
            if tag not in self.show_tags:
                continue
            # Draw the bbox as a rectangle
            ax.add_patch(plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color=c,
                linewidth=2
            ))
            # Draw the label
            text = ""
            if self.show_tag:
                text += f'{tag}:" '
            if self.show_confidence:
                text += f'{p[cl]:0.2f} '
            ax.text(xmin, ymin, text, fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5))

        plt.axis('off')
        if self.IMG_OUT:
            if not os.path.exists(self.IMG_OUT):
                os.makedirs(self.IMG_OUT)

            file_name_jpg = os.path.join(self.IMG_OUT, datetime.today().strftime("%Y-%m-%d-%H-%M-%S") + ".jpg")
            plt.savefig(file_name_jpg)
            print(f"Image Saved: {file_name_jpg}")
        if self.show_plot:
            plt.show()

