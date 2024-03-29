import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import error as err
import datetime


def plot_hist(subplot, data, rng=5, delta=None):
    if delta is None:
        delta = 0.3 / (len(data) / 512)
    a = int(rng // delta + 1)
    bins = [(x - a / 2) * delta for x in range(a)]
    hist, bin_edges = np.histogram(data, bins=bins)

    subplot.hist(data, bins=bin_edges)


def plot_results(real, fake, title):
    figure = plt.figure()
    figure.suptitle(title)
    for i in range(6):
        subplot = figure.add_subplot(2, 3, i + 1)
        # subplot.set_ylim([0, 1])
        subplot.set_title(f'Station {i + 1}')
        sb.histplot([x[i] for x in real], kde=True, stat="density")
        sb.histplot([x[i] for x in fake], kde=True, stat="density")

    figure.tight_layout()

    plt.show()


def load_data(file):
    columns = ["s1", "s2", "s3", "s4", "s5", "s6"]
    df = pd.read_csv(file, usecols=columns)
    # df = (df - df.mean()) / df.std()
    # df = (df - df.min()) / (df.max() - df.min())

    data = np.asarray(df.values.tolist())

    return data


def compare(model, file, num_samples, offset):
    data = load_data(file)

    real_data = data[offset:num_samples + offset, :]
    model.eval()
    generated_data = []
    for i in range(num_samples):
        generated_data.append(model.generate_sample().detach().cpu().tolist())

    print('Plotting results...')
    plot_results(real_data, generated_data, f'{model.name}-e{model.current_epoch}')

    print('Calculating marginal error...')
    marginal_error = err.anderson_darling(real_data, generated_data)
    print(f'Marginal error is {marginal_error}')


def compare_cvae(model, file, num_samples, offset, calc_ad=True, calc_dep=True):
    stations = [-3.242, -11.375,
                -4.992, -0.425,
                -0.292, 2.875,
                7.758, 3.525,
                1.608, -6.025,
                -0.842, 11.425]

    data = load_data(file)

    num_samples = num_samples if num_samples > 0 else len(data) - offset
    real_data = data[offset:num_samples + offset, :]
    model.eval()
    generated_data = np.zeros((num_samples, 6))
    for i in range(num_samples):
        for s in range(6):
            noise = np.random.randn(50)
            position = stations[s*2:(s+1)*2]
            generated_data[i, s] = model.generate_sample(noise, position)

    print('Plotting results...')
    plot_results(real_data, generated_data, f'{model.name}-e{model.current_epoch}')

    model.save()

    if calc_ad:
        print('Calculating marginal error...')
        marginal_error = err.anderson_darling(real_data, generated_data)
        print(f'Marginal error is {marginal_error}')

    if calc_dep:
        print('Calculating dependency error...')
        dependency_error = err.kendall(real_data, generated_data)
        print(dependency_error)


def path_from_name(name):
    date_time_str = datetime.now().strftime("%Y-%m-%d %H;%M;%S")
    return f'saved\\{date_time_str}\\{name}.params'
