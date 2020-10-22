import torch
import torch.distributions as dists
from torch.utils.data import Dataset


def generate_data_additive_treatment_process(m, alpha=1, print_details=False):
    normal_sigma_1 = dists.Normal(0, 1)
    bern = dists.Bernoulli(0.5)
    unif = dists.Uniform(0, 1)

    z = normal_sigma_1.sample((m,))
    eps = normal_sigma_1.sample((m,))

    assert z.shape == eps.shape, (z.shape, eps.shape)
    t = (z + eps)/1.414

    if print_details:
        print("generated data with treated fraction {}".format(torch.mean(t)))
        print("running with alpha = ", alpha)
    y = 0.32*normal_sigma_1.sample((m,)) + t + alpha*t*t*z

    # replace 0*z with just z to use the confounder in the main code
    return None, 0*z, eps, t, y

# def generate_data_mulnoiseT_experiment(m, alpha=1):


def generate_data_multiplicative_treatment_process(m, alpha=1, print_details=False):
    normal_sigma_1 = dists.Normal(0, 1)
    bern = dists.Bernoulli(0.5)
    unif = dists.Uniform(0, 1)

    z = normal_sigma_1.sample((m,))
    eps = normal_sigma_1.sample((m,))

    assert z.shape == eps.shape, (z.shape, eps.shape)
    t = z*eps

    y = 0.32*normal_sigma_1.sample((m,)) + t + alpha*z

    # replace 0*z with just z to use the confounder in the main code
    return None, 0*z, eps, t, y


def generate_mixed_data(m, alpha, frac):
    normal_sigma_1 = dists.Normal(0, 1)
    bern = dists.Bernoulli(frac)
    unif = dists.Uniform(0, 1)

    z = normal_sigma_1.sample((m,))
    eps = normal_sigma_1.sample((m,))
    t = z*eps

    y = 0.32*normal_sigma_1.sample((m,)) + t + alpha*t*z
    mask = bern.sample((m,))

    print('samples have {} observed z'.format(mask.mean()))

    return None, mask*z, eps, t, y


def indexes_to_one_hot(indexes, n_dims):
    """Converts a vector of indexes to a batch of one-hot vectors. """
    indexes = indexes.type(torch.int64).view(-1, 1)
    one_hots = torch.zeros(indexes.size()[0], n_dims).scatter_(1, indexes, 1)
    one_hots = one_hots.view(*indexes.shape, -1)
    return one_hots


def real_number_batch_to_one_hot_vector_bins(real_numbers, bin_count, return_one_hot=False, r_min=-3.5, r_max=3.5):
    """Converts a batch of real numbers to a batch of one hot vectors for the bins the real numbers fall in."""
    # r_min and r_max change depending on the data;
    # TODO: Is there a better way to handling the range?

    bins = torch.arange(1e-4 + r_min, r_max + 1e-4, (r_max - r_min)/bin_count)
    try:
        _, indexes = (real_numbers.view(-1, 1) -
                      bins.view(1, -1)).abs().min(dim=1)
    except:
        assert False, (real_numbers.type(), bins.type())
    if not return_one_hot:
        return indexes
    else:
        return indexes_to_one_hot(indexes, n_dims=bin_count).view(-1, bin_count)


class XZETY_Dataset(Dataset):
    def __init__(self, z, eps, t, y, x=None):
        """dataset for vde

        Args:
            x ([type]): covariates
            z ([type]): confounder
            eps ([type]): IV
            t ([type]): treatment
            y ([type]): outcome
        """
        self.x = x
        self.z = z
        self.eps = eps
        self.t = t
        self.y = y

        if x is not None:
            assert x.shape[0] == t.shape[0]
        assert eps.shape[0] == t.shape[0]
        assert z.shape[0] == t.shape[0]
        assert y.shape[0] == t.shape[0]

    def __len__(self):
        return self.t.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (
            self.x[idx] if self.x is not None else torch.Tensor(),
            self.z[idx],
            self.eps[idx],
            self.t[idx],
            self.y[idx]
        )

        return sample
