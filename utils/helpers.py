import os
import time
import sys
import numpy as np
import shutil
import random
import glob2


def get_size_dataset(dir_path):
    count = 0

    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    return count


def draw_curve(dir_save_fig, current_epoch, x_epoch, y_loss, y_err, fig, ax0, ax1):
    x_epoch.append(current_epoch + 1)
    ax0.plot(x_epoch, y_loss['train'], 'b-', linewidth=1.0, label='train')
    ax0.plot(x_epoch, y_loss['val'], '-r', linewidth=1.0, label='val')
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("loss")
    ax1.plot(x_epoch, y_err['train'], '-b', linewidth=1.0, label='train')
    ax1.plot(x_epoch, y_err['val'], '-r', linewidth=1.0, label='val')
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("error")
    if current_epoch == 0:
        ax0.legend(loc="upper right")
        ax1.legend(loc="upper right")
    fig.savefig(os.path.join(dir_save_fig, 'train_curves.jpg'), dpi=600)


class Kbar(object):
    """Keras progress bar.
    Arguments:
            target: Total number of steps expected, None if unknown.
            epoch: Zeor-indexed current epoch.
            num_epochs: Total epochs.
            width: Progress bar width on screen.
            verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
            always_stateful: (Boolean) Whether to set all metrics to be stateful.
            stateful_metrics: Iterable of string names of metrics that
                    should *not* be averaged over time. Metrics in this list
                    will be displayed as-is. All others will be averaged
                    by the progbar before display.
            interval: Minimum visual progress update interval (in seconds).
            unit_name: Display name for step counts (usually "step" or "sample").
    """

    def __init__(self, target, epoch=None, num_epochs=None,
                 width=30, verbose=1, interval=0.05,
                 stateful_metrics=None, always_stateful=False,
                 unit_name='step'):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.unit_name = unit_name
        self.always_stateful = always_stateful
        if (epoch is not None) and (num_epochs is not None):
            print('Epoch: %d/%d' % (epoch + 1, num_epochs))
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty')
                                  and sys.stdout.isatty())
                                 or 'ipykernel' in sys.modules
                                 or 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        Arguments:
                current: Index of current step.
                values: List of tuples:
                        `(name, value_for_last_step)`.
                        If `name` is in `stateful_metrics`,
                        `value_for_last_step` will be displayed as-is.
                        Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            # if torch tensor, convert it to numpy
            if str(type(v)) == "<class 'torch.Tensor'>":
                v = v.detach().cpu().numpy()

            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics and not self.always_stateful:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value. This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval
                    and self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                bar = ('%' + str(numdigits) + 'd/%d [') % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1 or time_per_unit == 0:
                    info += ' %.0fs/%s' % (time_per_unit, self.unit_name)
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/%s' % (time_per_unit * 1e3, self.unit_name)
                else:
                    info += ' %.0fus/%s' % (time_per_unit * 1e6, self.unit_name)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is not None and current >= self.target:
                numdigits = int(np.log10(self.target)) + 1
                count = ('%' + str(numdigits) + 'd/%d') % (current, self.target)
                info = count + info
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


class Pbar(object):
    """ Progress bar with title and timer
    Arguments:
    name: the bars name.
    target: Total number of steps expected.
    width: Progress bar width on screen.
    Usage example
    ```
    import kpbar
    import time
    pbar = kpbar.Pbar('loading and processing dataset', 10)
    for i in range(10):
        time.sleep(0.1)
        pbar.update(i)
    ```
    ```output
    loading and processing dataset
    10/10  [==============================] - 1.0s
    ```
    """

    def __init__(self, name, target, width=30):
        self.name = name
        self.target = target
        self.start = time.time()
        self.numdigits = int(np.log10(self.target)) + 1
        self.width = width
        print(self.name)

    def update(self, step):

        bar = ('%' + str(self.numdigits) + 'd/%d ') % (step + 1, self.target)

        status = ""

        if step < 0:
            step = 0
            status = "negtive?...\r\n"

        stop = time.time()

        status = '- {:.1f}s'.format((stop - self.start))

        progress = float(step + 1) / self.target

        # prog
        prog_width = int(self.width * progress)
        prog = ''
        if prog_width > 0:
            prog += ('=' * (prog_width - 1))
            if step + 1 < self.target:
                prog += '>'
            else:
                prog += '='
        prog += ('.' * (self.width - prog_width))

        # text = "\r{0} {1} [{2}] {3:.0f}% {4}".format(self.name, bar, prog, pregress, status)

        text = "\r{0} [{1}] {2}".format(bar, prog, status)
        sys.stdout.write(text)
        if step + 1 == self.target:
            sys.stdout.write('\n')
        sys.stdout.flush()


def split_fold(num_fold=10, test_image_number=380):
    print("Splitting for k-fold with %d fold" % num_fold)
    data_root = os.path.join(os.getcwd(), 'data')

    dir_names = []
    for fold in range(num_fold):
        dir_names.append('data/TrainData' + str(fold))

    for dir_name in dir_names:
        print("Creating fold " + dir_name)
        os.makedirs(dir_name)

        # making subdirectory train and test
        os.makedirs(os.path.join(os.getcwd(), dir_name, 'test'))
        os.makedirs(os.path.join(os.getcwd(), dir_name, 'train'))

        # locating to the test and train directory
        test_dir = os.path.join(os.getcwd(), dir_name, 'test')
        train_dir = os.path.join(os.getcwd(), dir_name, 'train')

        # making image and mask sub-dirs
        os.makedirs(os.path.join(test_dir, 'img'))
        os.makedirs(os.path.join(test_dir, 'mask'))
        os.makedirs(os.path.join(train_dir, 'img'))
        os.makedirs(os.path.join(train_dir, 'mask'))

        # read the image and mask directory
        image_files = os.listdir(os.path.join(os.getcwd(), 'data/img'))
        mask_files = os.listdir(os.path.join(os.getcwd(), 'data/mask'))

        # creating random file names for testing
        test_filenames = random.sample(image_files, test_image_number)

        for filename in test_filenames:
            img_data_root = os.path.join(data_root, 'img')
            msk_data_root = os.path.join(data_root, 'mask')

            img_dest = os.path.join(os.getcwd(), dir_name, 'test', 'img')
            msk_dest = os.path.join(os.getcwd(), dir_name, 'test', 'mask')

            img_file_path = os.path.join(img_data_root, filename)
            msk_file_path = os.path.join(msk_data_root, filename.replace('image', 'mask'))

            shutil.copy(img_file_path, img_dest)
            shutil.copy(msk_file_path, msk_dest)

        # saving files for training
        for other_filename in image_files:
            if other_filename in test_filenames:
                continue
            else:
                img_data_root = os.path.join(data_root, 'img')
                msk_data_root = os.path.join(data_root, 'mask')

                img_dest = os.path.join(os.getcwd(), dir_name, 'train', 'img')
                msk_dest = os.path.join(os.getcwd(), dir_name, 'train', 'mask')

                img_file_path = os.path.join(img_data_root, other_filename)
                msk_file_path = os.path.join(msk_data_root, other_filename.replace('image', 'mask'))

                shutil.copy(img_file_path, img_dest)
                shutil.copy(msk_file_path, msk_dest)


def get_train_list(fold):
    all_files = []
    for ext in ["*.h5"]:
        images = glob2.glob(os.path.join("data/TrainData" + str(fold) + "/train/img/", ext))
        all_files += images

    all_train_files = []
    for idx in np.arange(len(all_files)):
        image = str(all_files[idx]).split("/")
        image = os.path.join("TrainData" + str(fold) + "/train/img/", str(image[4]))
        all_train_files.append(image)

    # Create train.txt
    with open("dataset/train.txt", "w") as f:
        for idx in np.arange(len(all_train_files)):
            f.write(all_train_files[idx] + '\n')


def get_test_list(fold):
    all_files = []
    for ext in ["*.h5"]:
        images = glob2.glob(os.path.join("data/TrainData" + str(fold) + "/test/img/", ext))
        all_files += images

    all_test_files = []
    for idx in np.arange(len(all_files)):
        image = str(all_files[idx]).split("/")
        image = os.path.join("TrainData" + str(fold) + "/test/img/", str(image[4]))
        all_test_files.append(image)

    # Create Test.txt
    with open("dataset/test.txt", "w") as f:
        for idx in np.arange(len(all_test_files)):
            f.write(all_test_files[idx] + '\n')


def get_train_test_list(fold):
    get_train_list(fold)
    get_test_list(fold)
