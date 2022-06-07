import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def display_image(image):
    plt.imshow(image, cmap="Greys")
    plt.show()

class NN:
    def __init__(self, in_width, in_height, filter_width, filter_height, no_filters, pool_width, pool_height, output_size):
        self.input_size = (in_width, in_height)
        self.l1_size = (no_filters, in_height - filter_height + 1, in_width - filter_width + 1)

        assert (self.l1_size[2] % pool_width) == 0
        assert (self.l1_size[1] % pool_height) == 0
        
        self.p_size = (no_filters, self.l1_size[1]//pool_height, self.l1_size[2]//pool_width)
        self.filters = np.zeros([no_filters, filter_height, filter_width])
        self.filter_width = filter_width 
        self.filter_height = filter_height
        self.no_filters = no_filters
        self.pool_width = pool_width
        self.pool_height = pool_height
        self.weight_length = self.p_size[0] * self.p_size[1] * self.p_size[2]
        self.weights = np.zeros([output_size, self.weight_length])
        self.bias = np.ones([output_size])
        self.output_size = output_size

    def load_model(self, path):
        with open(path, "r") as file:
            raw = file.read()
            nn_values = json.loads(raw)
            self.filters = np.array(nn_values["filters"])
            self.weights = np.array(nn_values["weights"])
            self.bias = np.array(nn_values["bias"])
            self.p_size = nn_values["p_size"]
            self.filter_width = nn_values["filter_width"]
            self.filter_height = nn_values["filter_height"]
            self.no_filters = nn_values["no_filters"]
            self.pool_width = nn_values["pool_width"]
            self.pool_height = nn_values["pool_height"]
            self.weight_length = nn_values["weight_length"]
            self.output_size = nn_values["output_size"]

    def save_model(self, path):
        with open(path, "w") as file:
            nn_values = dict()
            nn_values["filters"] = self.filters.tolist()
            nn_values["weights"] = self.weights.tolist()
            nn_values["bias"] = self.bias.tolist()
            nn_values["p_size"] = self.p_size 
            nn_values["filter_width"] = self.filter_width 
            nn_values["filter_height"] = self.filter_height 
            nn_values["no_filters"] = self.no_filters
            nn_values["pool_width"] = self.pool_width 
            nn_values["pool_height"] = self.pool_height
            nn_values["weight_length"] = self.weight_length
            nn_values["output_size"] = self.output_size
            raw = json.dumps(nn_values)
            file.write(raw)

    def set_filters(self, filters):
        assert len(filters) == self.no_filters
        assert len(filters[0]) == self.filter_height
        assert len(filters[0,0]) == self.filter_width
        self.filters = filters

    def xavier_init(self):
        weights = np.random.randn(self.output_size, self.weight_length) / self.weight_length
        self.weights = weights
        return weights

    def convolute(self, image):
        assert len(image) == self.input_size[0]
        assert len(image[0]) == self.input_size[1]

        first_layer = np.zeros(self.l1_size)
        for i, filter in enumerate(self.filters):
            for j in range(self.l1_size[1]):
                for k in range(self.l1_size[2]):
                    t = np.tensordot(image[j:j+self.filter_width, k:k+self.filter_height], filter)
                    first_layer[i,j,k] = t
                    
        self.l1_cache = first_layer[:]
        return first_layer

    def pool(self, first_layer):
        pooled = np.zeros(self.p_size)
        for i in range(self.no_filters):
            for j in range(0, self.l1_size[1], self.pool_width):
                for k in range(0, self.l1_size[2], self.pool_height):
                    pooled[i,j//self.pool_width,k//self.pool_height] = first_layer[i,j:j+self.pool_height,k:k+self.pool_width].max()
        self.p_cache = pooled
        return pooled

    def ff(self, ff_in):
        ff_out = np.zeros(self.output_size)
        flat = ff_in.flatten()
        self.flat_cache = flat
        for i in range(self.output_size):
            ff_out[i] = np.dot(self.weights[i], flat) + self.bias[i]
        self.ff_cache = ff_out
        return ff_out

    def soft_max(self, ff_out):
        exps = np.exp(ff_out.astype(np.longdouble))
        exp_sum = np.sum(exps)
        self.output = exps/exp_sum
        return self.output

    def feed_forward(self, image, debug=False):
        if debug:
            display_image(image)
        l1 = self.convolute(image)
        if debug:
            for i in l1:
                print(i)
                display_image(i)
        p = self.pool(l1)
        if debug:
            for i in p:
                display_image(i)
        ff_out = self.ff(p)
        if debug:
            print(ff_out)
        predict = self.soft_max(ff_out)
        if debug:
            print(predict)
        return predict

    # dL_dout, change in loss in relation to the change in output
    def xel_backprop(self, predict, correct):
        dL_dout = np.zeros([self.output_size])
        dL_dout[correct] = -1/predict[correct]
        self.dL_dout_cache = dL_dout
        return dL_dout

    # dout_dt, change in output in relation to change in pre_softmax totals
    def soft_max_backprop(self, ts, c):
        dout_dt = np.zeros([self.output_size])
        t_exp = np.exp(ts.astype(np.longdouble))
        S = np.sum(t_exp)

        dout_dt = -t_exp[c] * t_exp / (S ** 2)
        dout_dt[c] = t_exp[c] * (S - t_exp[c]) / (S ** 2)
        
        return dout_dt

    def ff_backprop(self, dL_dout, dout_dt):
        dt_dw = self.flat_cache # 676 x 1
        dt_db = 1
        dt_dflat = self.weights # 10 x 676
        dL_dt = np.multiply(dL_dout, dout_dt)  # 10 x 1
        
        dL_dw = np.zeros(self.weights.shape)  # 10 x 676 
        dL_db = np.zeros(self.output_size)  # 10 x 1
        dL_dflat = np.zeros(self.flat_cache.shape)  # 676 x 1

        dL_db = dL_dt * dt_db
        dL_dflat = dL_dt @ dt_dflat
        dL_dw = np.outer(dL_dt, dt_dw)
        return (dL_db, dL_dflat, dL_dw)

    def pool_backprop(self, dL_dflat):
        dL_da1 = np.zeros(self.l1_cache.shape)
        for i in range(self.no_filters):
            for j in range(0, self.l1_size[1], self.pool_width):
                for k in range(0, self.l1_size[2], self.pool_height):
                    max_pos = self.l1_cache[i, j:j+self.pool_height,k:k+self.pool_width].argmax()
                    max_y = j + max_pos//self.pool_height
                    max_x = k + max_pos%self.pool_width
                    dL_da1[i, max_y, max_x] = dL_dflat[i+j+k]
        return dL_da1

    def conv_backprop(self, image, dL_da1):
        dL_dfilters = np.zeros(self.filters.shape)
        for i in range(self.no_filters):
            for y in range(self.filter_height):
                for x in range(self.filter_width):
                    dL_dfilters[i,y,x] = (
                    image[
                        y:self.input_size[1] - self.filter_height + y + 1, 
                        x:self.input_size[0] - self.filter_width + x + 1
                    ] * dL_da1[i] ).sum()
        return dL_dfilters

    def back_prop(self, image, predict, correct, rate):
        dL_dout = self.xel_backprop(predict, correct)
        dout_dt = self.soft_max_backprop(self.ff_cache, correct)
        dL_db, dL_dflat, dL_dw = self.ff_backprop(dL_dout, dout_dt)
        self.dL_dw_cache = dL_dw
        self.dL_db_cache = dL_db
        self.weights -= rate * dL_dw
        self.bias -= rate * dL_db
        dL_da1 = self.pool_backprop(dL_dflat)
        dL_dfilters = self.conv_backprop(image, dL_da1)
        self.filters -= rate * dL_dfilters

    def train(self, images, labels, rate, early_stop = 0):
        total, epoch_correct, epoch_loss = 0, 0, 0
        for i, (image, label) in enumerate(zip(images, labels)):
            total += 1
            predict = self.feed_forward(image)
            if np.argmax(predict) == label:
                epoch_correct += 1
            epoch_loss += -np.log(predict[label])
            self.back_prop(image, predict, label, rate)


            if i % 100 == 99:
                print(f"[Step {total}] Past 100 steps: Avg. Loss {epoch_loss/100} | Accuracy: {epoch_correct}%")
                epoch_correct = 0
                if epoch_loss < early_stop:
                    print(epoch_loss)
                    return
                epoch_loss = 0

    def test(self, images, labels):
        total, total_correct, epoch_correct, epoch_loss, total_loss = 0, 0, 0, 0, 0
        digit_acc = [0] * 10
        for i, (image, label) in enumerate(zip(images, labels)):
            total += 1
            predict = self.feed_forward(image)
            if np.argmax(predict) == label:
                epoch_correct += 1
                total_correct += 1
                digit_acc[int(label)] += 1
            loss = -np.log(predict[int(label)])
            epoch_loss += loss
            total_loss += loss

            if i % 100 == 99:
                print(f"[Step {total}] Past 100 steps: Avg. Loss {epoch_loss/100} | Accuracy: {epoch_correct}%")
                epoch_correct = 0
                epoch_loss = 0
        print(f"[Test Results] All 10000 test steps: Avg. Loss {total_loss/len(images)} | Accuracy: {100*total_correct/len(images):.2f}%")
        
        names, totals = np.unique(labels, return_counts=True)
        values = digit_acc/totals
        sns.set_theme()
        ax = sns.barplot(names, values)
        ax.bar_label(ax.containers[0])
        plt.ylim(0,1)
        plt.title("Accuracy of prediction of each digit")
        plt.show()
