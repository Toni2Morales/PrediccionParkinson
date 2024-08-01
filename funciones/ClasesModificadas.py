# from sdv.single_table import CTGANSynthesizer
# from sdv.single_table.ctgan import _validate_no_category_dtype
# from sdv.single_table.utils import detect_discrete_columns
import numpy as np
import pandas as pd
import logging
import sys
from copy import deepcopy
from rdt.transformers import OneHotEncoder
# from ctgan import CTGAN
# from ctgan.synthesizers.base import random_state
# from ctgan.data_transformer import DataTransformer
# from ctgan.data_sampler import DataSampler
# from ctgan.synthesizers.ctgan import Generator, Discriminator
from sdv.single_table.base import BaseSingleTableSynthesizer
from sdv.single_table.ctgan import LossValuesMixin
from sdv.single_table.utils import (
    flatten_dict, log_numerical_distributions_error, unflatten_dict,
    validate_numerical_distributions)
from sdv.errors import NonParametricError
import copulas
from copulas import (
    EPSILON, check_valid_values, get_instance, get_qualified_name, random_state, store_args,
    validate_random_state)
from copulas.multivariate.base import Multivariate
from copulas.univariate import GaussianUnivariate, Univariate
LOGGER = logging.getLogger(__name__)
DEFAULT_DISTRIBUTION = Univariate
from copulas import multivariate
import copulas.univariate
import scipy
from scipy import stats
import inspect
# from torch import optim
# import torch
# from tqdm import tqdm
import warnings
LOGGER = logging.getLogger(__name__)
# class CTGANMod(CTGAN):
#     def __init__(self, EarlyStoppingEspera, EarlyStoppingDisminucion,
#                  embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
#                  generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
#                  discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
#                  log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True):
#         super().__init__(embedding_dim=embedding_dim, generator_dim=generator_dim, discriminator_dim=discriminator_dim,
#                  generator_lr=generator_lr, generator_decay=generator_decay, discriminator_lr=discriminator_lr,
#                  discriminator_decay=discriminator_decay, batch_size=batch_size, discriminator_steps=discriminator_steps,
#                  log_frequency=log_frequency, verbose=verbose, epochs=epochs, pac=pac, cuda=cuda)
#         self.EarlyStoppingEspera = EarlyStoppingEspera #La espera es el número de épocas para esperar antes de detener el entrenamiento.
#         self.EarlyStoppingDisminucion = EarlyStoppingDisminucion #Mínima disminución requerida en la pérdida para continuar el entrenamiento.
#         self.EarlyStoppingMejor = None
#         self.NumMedia = 50
#     def EarlyStopping(self, Loss): # Función para aplicar early stopping al entrenamiento
#         if self.EarlyStoppingMejor == None:
#             self.EarlyStoppingMejor = np.mean(np.abs(Loss))
#         else:
#             if np.mean(np.abs(Loss[:-(self.NumMedia+1):-1])) + self.EarlyStoppingDisminucion < self.EarlyStoppingMejor:
#                 self.EarlyStoppingMejor = np.mean(np.abs(Loss[:-(self.NumMedia+1):-1]))
#                 return True
#             else:
#                 return False
#     @random_state
#     def fit(self, train_data, discrete_columns=(), epochs=None):
#         """Fit the CTGAN Synthesizer models to the training data.

#         Args:
#             train_data (numpy.ndarray or pandas.DataFrame):
#                 Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
#             discrete_columns (list-like):
#                 List of discrete columns to be used to generate the Conditional
#                 Vector. If ``train_data`` is a Numpy array, this list should
#                 contain the integer indices of the columns. Otherwise, if it is
#                 a ``pandas.DataFrame``, this list should contain the column names.
#         """
#         self._validate_discrete_columns(train_data, discrete_columns)

#         if epochs is None:
#             epochs = self._epochs
#         else:
#             warnings.warn(
#                 ('`epochs` argument in `fit` method has been deprecated and will be removed '
#                  'in a future version. Please pass `epochs` to the constructor instead'),
#                 DeprecationWarning
#             )

#         self._transformer = DataTransformer()
#         self._transformer.fit(train_data, discrete_columns)

#         train_data = self._transformer.transform(train_data)

#         self._data_sampler = DataSampler(
#             train_data,
#             self._transformer.output_info_list,
#             self._log_frequency)

#         data_dim = self._transformer.output_dimensions

#         self._generator = Generator(
#             self._embedding_dim + self._data_sampler.dim_cond_vec(),
#             self._generator_dim,
#             data_dim
#         ).to(self._device)

#         discriminator = Discriminator(
#             data_dim + self._data_sampler.dim_cond_vec(),
#             self._discriminator_dim,
#             pac=self.pac
#         ).to(self._device)

#         optimizerG = optim.Adam(
#             self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
#             weight_decay=self._generator_decay
#         )

#         optimizerD = optim.Adam(
#             discriminator.parameters(), lr=self._discriminator_lr,
#             betas=(0.5, 0.9), weight_decay=self._discriminator_decay
#         )

#         mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
#         std = mean + 1

#         self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

#         epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
#         if self._verbose:
#             description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
#             epoch_iterator.set_description(description.format(gen=0, dis=0))

#         steps_per_epoch = max(len(train_data) // self._batch_size, 1)
#         Contador = 0
#         for i in epoch_iterator:
#             for id_ in range(steps_per_epoch):
#                 for n in range(self._discriminator_steps):
#                     fakez = torch.normal(mean=mean, std=std)

#                     condvec = self._data_sampler.sample_condvec(self._batch_size)
#                     if condvec is None:
#                         c1, m1, col, opt = None, None, None, None
#                         real = self._data_sampler.sample_data(
#                             train_data, self._batch_size, col, opt)
#                     else:
#                         c1, m1, col, opt = condvec
#                         c1 = torch.from_numpy(c1).to(self._device)
#                         m1 = torch.from_numpy(m1).to(self._device)
#                         fakez = torch.cat([fakez, c1], dim=1)

#                         perm = np.arange(self._batch_size)
#                         np.random.shuffle(perm)
#                         real = self._data_sampler.sample_data(
#                             train_data, self._batch_size, col[perm], opt[perm])
#                         c2 = c1[perm]

#                     fake = self._generator(fakez)
#                     fakeact = self._apply_activate(fake)

#                     real = torch.from_numpy(real.astype('float32')).to(self._device)

#                     if c1 is not None:
#                         fake_cat = torch.cat([fakeact, c1], dim=1)
#                         real_cat = torch.cat([real, c2], dim=1)
#                     else:
#                         real_cat = real
#                         fake_cat = fakeact

#                     y_fake = discriminator(fake_cat)
#                     y_real = discriminator(real_cat)

#                     pen = discriminator.calc_gradient_penalty(
#                         real_cat, fake_cat, self._device, self.pac)
#                     loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

#                     optimizerD.zero_grad(set_to_none=False)
#                     pen.backward(retain_graph=True)
#                     loss_d.backward()
#                     optimizerD.step()

#                 fakez = torch.normal(mean=mean, std=std)
#                 condvec = self._data_sampler.sample_condvec(self._batch_size)

#                 if condvec is None:
#                     c1, m1, col, opt = None, None, None, None
#                 else:
#                     c1, m1, col, opt = condvec
#                     c1 = torch.from_numpy(c1).to(self._device)
#                     m1 = torch.from_numpy(m1).to(self._device)
#                     fakez = torch.cat([fakez, c1], dim=1)

#                 fake = self._generator(fakez)
#                 fakeact = self._apply_activate(fake)

#                 if c1 is not None:
#                     y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
#                 else:
#                     y_fake = discriminator(fakeact)

#                 if condvec is None:
#                     cross_entropy = 0
#                 else:
#                     cross_entropy = self._cond_loss(fake, c1, m1)

#                 loss_g = -torch.mean(y_fake) + cross_entropy

#                 optimizerG.zero_grad(set_to_none=False)
#                 loss_g.backward()
#                 optimizerG.step()

#             generator_loss = loss_g.detach().cpu().item()
#             discriminator_loss = loss_d.detach().cpu().item()

#             epoch_loss_df = pd.DataFrame({
#                 'Epoch': [i],
#                 'Generator Loss': [generator_loss],
#                 'Discriminator Loss': [discriminator_loss]
#             })
#             if not self.loss_values.empty:
#                 self.loss_values = pd.concat(
#                     [self.loss_values, epoch_loss_df]
#                 ).reset_index(drop=True)
#             else:
#                 self.loss_values = epoch_loss_df
#             if self._verbose:
#                 epoch_iterator.set_description(
#                     description.format(gen=generator_loss, dis=discriminator_loss)
#                 )
#                 #Esta es nuestra modificación
#             if i >= self.NumMedia -1:
#                 GenLoss = self.EarlyStopping(self.loss_values["Generator Loss"])
#                 DisLoss = self.EarlyStopping(self.loss_values["Discriminator Loss"])
#                 if not(GenLoss) and not(DisLoss):
#                     Contador += 1
#                     if (Contador >= self.EarlyStoppingEspera) and i >= 500:
#                         print("Se detuvo el entrenamiento prematuramente debido a la falta de mejora.")
#                         break
#                 else:
#                     Contador = 0
# class CTGANSynthesizerMod(CTGANSynthesizer):
#     """Model wrapping ``CTGAN`` model.

#     Args:
#         metadata (sdv.metadata.SingleTableMetadata):
#             Single table metadata representing the data that this synthesizer will be used for.
#         enforce_min_max_values (bool):
#             Specify whether or not to clip the data returned by ``reverse_transform`` of
#             the numerical transformer, ``FloatFormatter``, to the min and max values seen
#             during ``fit``. Defaults to ``True``.
#         enforce_rounding (bool):
#             Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
#             by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
#         locales (list or str):
#             The default locale(s) to use for AnonymizedFaker transformers.
#             Defaults to ``['en_US']``.
#         embedding_dim (int):
#             Size of the random sample passed to the Generator. Defaults to 128.
#         generator_dim (tuple or list of ints):
#             Size of the output samples for each one of the Residuals. A Residual Layer
#             will be created for each one of the values provided. Defaults to (256, 256).
#         discriminator_dim (tuple or list of ints):
#             Size of the output samples for each one of the Discriminator Layers. A Linear Layer
#             will be created for each one of the values provided. Defaults to (256, 256).
#         generator_lr (float):
#             Learning rate for the generator. Defaults to 2e-4.
#         generator_decay (float):
#             Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
#         discriminator_lr (float):
#             Learning rate for the discriminator. Defaults to 2e-4.
#         discriminator_decay (float):
#             Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
#         batch_size (int):
#             Number of data samples to process in each step.
#         discriminator_steps (int):
#             Number of discriminator updates to do for each generator update.
#             From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
#             default is 5. Default used is 1 to match original CTGAN implementation.
#         log_frequency (boolean):
#             Whether to use log frequency of categorical levels in conditional
#             sampling. Defaults to ``True``.
#         verbose (boolean):
#             Whether to have print statements for progress results. Defaults to ``False``.
#         epochs (int):
#             Number of training epochs. Defaults to 300.
#         pac (int):
#             Number of samples to group together when applying the discriminator.
#             Defaults to 10.
#         cuda (bool or str):
#             If ``True``, use CUDA. If a ``str``, use the indicated device.
#             If ``False``, do not use cuda at all.
#     """

#     _model_sdtype_transformers = {'categorical': None, 'boolean': None}

#     def __init__(
#         self,
#         metadata,
#         enforce_min_max_values=True,
#         enforce_rounding=True,
#         locales=['en_US'],
#         embedding_dim=128,
#         generator_dim=(256, 256),
#         discriminator_dim=(256, 256),
#         generator_lr=2e-4,
#         generator_decay=1e-6,
#         discriminator_lr=2e-4,
#         discriminator_decay=1e-6,
#         batch_size=500,
#         discriminator_steps=1,
#         log_frequency=True,
#         verbose=False,
#         epochs=300,
#         pac=10,
#         cuda=True,
#         EarlyStoppingEspera = 10,
#         EarlyStoppingDisminucion = 0.02
#     ):
#         super().__init__(
#         metadata,
#         enforce_min_max_values=enforce_min_max_values,
#         enforce_rounding=enforce_rounding,
#         locales=locales,
#         embedding_dim=embedding_dim,
#         generator_dim=generator_dim,
#         discriminator_dim=discriminator_dim,
#         generator_lr=generator_lr,
#         generator_decay=generator_decay,
#         discriminator_lr=discriminator_lr,
#         discriminator_decay=discriminator_decay,
#         batch_size=batch_size,
#         discriminator_steps=discriminator_steps,
#         log_frequency=log_frequency,
#         verbose=verbose,
#         epochs=epochs,
#         pac=pac,
#         cuda=cuda
#     )
#         self.EarlyStoppingEspera = EarlyStoppingEspera
#         self.EarlyStoppingDisminución = EarlyStoppingDisminucion

#         self._model_kwargs = {
#             'embedding_dim': embedding_dim,
#             'generator_dim': generator_dim,
#             'discriminator_dim': discriminator_dim,
#             'generator_lr': generator_lr,
#             'generator_decay': generator_decay,
#             'discriminator_lr': discriminator_lr,
#             'discriminator_decay': discriminator_decay,
#             'batch_size': batch_size,
#             'discriminator_steps': discriminator_steps,
#             'log_frequency': log_frequency,
#             'verbose': verbose,
#             'epochs': epochs,
#             'pac': pac,
#             'cuda': cuda,
#             "EarlyStoppingEspera": EarlyStoppingEspera,
#             "EarlyStoppingDisminucion": EarlyStoppingDisminucion
#         }

#     def _fit(self, processed_data):
#         """Fit the model to the table.

#         Args:
#             processed_data (pandas.DataFrame):
#                 Data to be learned.
#         """
#         _validate_no_category_dtype(processed_data)

#         transformers = self._data_processor._hyper_transformer.field_transformers
#         discrete_columns = detect_discrete_columns(
#             self.get_metadata(), processed_data, transformers
#         )
#         self._model = CTGANMod(**self._model_kwargs)
#         self._model.fit(processed_data, discrete_columns=discrete_columns)
class GaussianMultivariateMod(Multivariate):
    """Class for a multivariate distribution that uses the Gaussian copula.

    Args:
        distribution (str or dict):
            Fully qualified name of the class to be used for modeling the marginal
            distributions or a dictionary mapping column names to the fully qualified
            distribution names.
    """

    correlation = None
    columns = None
    univariates = None

    @store_args
    def __init__(self, distribution=DEFAULT_DISTRIBUTION, random_state=None):
        self.random_state = validate_random_state(random_state)
        self.distribution = distribution

    def __repr__(self):
        """Produce printable representation of the object."""
        if self.distribution == DEFAULT_DISTRIBUTION:
            distribution = ''
        elif isinstance(self.distribution, type):
            distribution = f'distribution="{self.distribution.__name__}"'
        else:
            distribution = f'distribution="{self.distribution}"'

        return f'GaussianMultivariate({distribution})'

    def _transform_to_normal(self, X):
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        elif not isinstance(X, pd.DataFrame):
            if len(X.shape) == 1:
                X = [X]

            X = pd.DataFrame(X, columns=self.columns)

        U = []
        for column_name, univariate in zip(self.columns, self.univariates):
            if column_name in X:
                column = X[column_name]
                U.append(univariate.cdf(column.to_numpy()).clip(EPSILON, 1 - EPSILON))

        return stats.norm.ppf(np.column_stack(U))

    def _get_correlation(self, X):
        """Compute correlation matrix with transformed data.

        Args:
            X (numpy.ndarray):
                Data for which the correlation needs to be computed.

        Returns:
            numpy.ndarray:
                computed correlation matrix.
        """
        result = self._transform_to_normal(X)
        correlation = pd.DataFrame(data=result).corr().to_numpy()
        correlation = np.nan_to_num(correlation, nan=0.0)
        # If singular, add some noise to the diagonal
        if np.linalg.cond(correlation) > 1.0 / sys.float_info.epsilon:
            correlation = correlation + np.identity(correlation.shape[0]) * EPSILON

        return pd.DataFrame(correlation, index=self.columns, columns=self.columns)

    @check_valid_values
    def fit(self, X):
        """Compute the distribution for each variable and then its correlation matrix.

        Arguments:
            X (pandas.DataFrame):
                Values of the random variables.
        """
        LOGGER.info('Fitting %s', self)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        columns = []
        univariates = []
        for column_name, column in X.items():
            if isinstance(self.distribution, dict):
                distribution = self.distribution.get(column_name, DEFAULT_DISTRIBUTION)
            else:
                distribution = self.distribution

            LOGGER.debug('Fitting column %s to %s', column_name, distribution)

            univariate = get_instance(distribution)
            try:
                univariate.fit(column)
            except BaseException:
                log_message = (
                    f'Unable to fit to a {distribution} distribution for column {column_name}. '
                    'Using a Gaussian distribution instead.'
                )
                LOGGER.info(log_message)
                univariate = GaussianUnivariate()
                univariate.fit(column)

            columns.append(column_name)
            univariates.append(univariate)

        self.columns = columns
        self.univariates = univariates

        LOGGER.debug('Computing correlation')
        self.correlation = self._get_correlation(X)
        self.fitted = True

        LOGGER.debug('GaussianMultivariate fitted successfully')

    def probability_density(self, X):
        """Compute the probability density for each point in X.

        Arguments:
            X (pandas.DataFrame):
                Values for which the probability density will be computed.

        Returns:
            numpy.ndarray:
                Probability density values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        transformed = self._transform_to_normal(X)

        return stats.multivariate_normal.pdf(
            transformed, cov=self.correlation, allow_singular=True)

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution value for each point in X.

        Arguments:
            X (pandas.DataFrame):
                Values for which the cumulative distribution will be computed.

        Returns:
            numpy.ndarray:
                Cumulative distribution values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        transformed = self._transform_to_normal(X)
        return stats.multivariate_normal.cdf(transformed, cov=self.correlation)

    def _get_conditional_distribution(self, conditions):
        """Compute the parameters of a conditional multivariate normal distribution.

        The parameters of the conditioned distribution are computed as specified here:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions

        Args:
            conditions (pandas.Series):
                Mapping of the column names and column values to condition on.
                The input values have already been transformed to their normal distribution.

        Returns:
            tuple:
                * means (numpy.array):
                    mean values to use for the conditioned multivariate normal.
                * covariance (numpy.array):
                    covariance matrix to use for the conditioned
                  multivariate normal.
                * columns (list):
                    names of the columns that will be sampled conditionally.
        """
        columns2 = conditions.index
        columns1 = self.correlation.columns.difference(columns2)

        sigma11 = self.correlation.loc[columns1, columns1].to_numpy()
        sigma12 = self.correlation.loc[columns1, columns2].to_numpy()
        sigma21 = self.correlation.loc[columns2, columns1].to_numpy()
        sigma22 = self.correlation.loc[columns2, columns2].to_numpy()

        mu1 = np.zeros(len(columns1))
        mu2 = np.zeros(len(columns2))

        sigma12sigma22inv = sigma12 @ np.linalg.inv(sigma22)

        mu_bar = mu1 + sigma12sigma22inv @ (conditions - mu2)
        sigma_bar = sigma11 - sigma12sigma22inv @ sigma21

        return mu_bar, sigma_bar, columns1

    def _get_normal_samples(self, num_rows, conditions):
        """Get random rows in the standard normal space.

        If no conditions are given, the values are sampled from a standard normal
        multivariate.

        If conditions are given, they are transformed to their equivalent standard
        normal values using their marginals and then the values are sampled from
        a standard normal multivariate conditioned on the given condition values.
        """
        if conditions is None:
            covariance = self.correlation
            columns = self.columns
            means = np.zeros(len(columns))
        else:
            conditions = pd.Series(conditions)
            normal_conditions = self._transform_to_normal(conditions)[0]
            normal_conditions = pd.Series(normal_conditions, index=conditions.index)
            means, covariance, columns = self._get_conditional_distribution(normal_conditions)

        samples = np.random.multivariate_normal(means, covariance, size=num_rows)
        return pd.DataFrame(samples, columns=columns)

    @random_state
    def sample(self, num_rows=1, conditions=None):
        """Sample values from this model.

        Argument:
            num_rows (int):
                Number of rows to sample.
            conditions (dict or pd.Series):
                Mapping of the column names and column values to condition on.

        Returns:
            numpy.ndarray:
                Array of shape (n_samples, *) with values randomly
                sampled from this model distribution. If conditions have been
                given, the output array also contains the corresponding columns
                populated with the given values.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()

        samples = self._get_normal_samples(num_rows, conditions)

        output = {}
        for column_name, univariate in zip(self.columns, self.univariates):
            if conditions and column_name in conditions:
                # Use the values that were given as conditions in the original space.
                output[column_name] = np.full(num_rows, conditions[column_name])
            else:
                cdf = stats.norm.cdf(samples[column_name])
                output[column_name] = univariate.percent_point(cdf)

        return pd.DataFrame(data=output)

    def to_dict(self):
        """Return a `dict` with the parameters to replicate this object.

        Returns:
            dict:
                Parameters of this distribution.
        """
        self.check_fit()
        univariates = [univariate.to_dict() for univariate in self.univariates]

        return {
            'correlation': self.correlation.to_numpy().tolist(),
            'univariates': univariates,
            'columns': self.columns,
            'type': get_qualified_name(self),
        }

    @classmethod
    def from_dict(cls, copula_dict):
        """Create a new instance from a parameters dictionary.

        Args:
            params (dict):
                Parameters of the distribution, in the same format as the one
                returned by the ``to_dict`` method.

        Returns:
            Multivariate:
                Instance of the distribution defined on the parameters.
        """
        instance = cls()
        instance.univariates = []
        columns = copula_dict['columns']
        instance.columns = columns

        for parameters in copula_dict['univariates']:
            instance.univariates.append(Univariate.from_dict(parameters))

        correlation = copula_dict['correlation']
        instance.correlation = pd.DataFrame(correlation, index=columns, columns=columns)
        instance.fitted = True

        return instance

class GaussianCopulaSynthesizerMod(LossValuesMixin, BaseSingleTableSynthesizer):
    """Model wrapping ``copulas.multivariate.GaussianMultivariate`` copula.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers.
            Defaults to ``['en_US']``.
        numerical_distributions (dict):
            Dictionary that maps field names from the table that is being modeled with
            the distribution that needs to be used. The distributions can be passed as either
            a ``copulas.univariate`` instance or as one of the following values:

                * ``norm``: Use a norm distribution.
                * ``beta``: Use a Beta distribution.
                * ``truncnorm``: Use a truncnorm distribution.
                * ``uniform``: Use a uniform distribution.
                * ``gamma``: Use a Gamma distribution.
                * ``gaussian_kde``: Use a GaussianKDE distribution. This model is non-parametric,
                  so using this will make ``get_parameters`` unusable.

        default_distribution (str):
            Copulas univariate distribution to use by default. Valid options are:

                * ``norm``: Use a norm distribution.
                * ``beta``: Use a Beta distribution.
                * ``truncnorm``: Use a Truncated Gaussian distribution.
                * ``uniform``: Use a uniform distribution.
                * ``gamma``: Use a Gamma distribution.
                * ``gaussian_kde``: Use a GaussianKDE distribution. This model is non-parametric,
                  so using this will make ``get_parameters`` unusable.
             Defaults to ``beta``.
    """

    _DISTRIBUTIONS = {
        'norm': copulas.univariate.GaussianUnivariate,
        'beta': copulas.univariate.BetaUnivariate,
        'truncnorm': copulas.univariate.TruncatedGaussian,
        'gamma': copulas.univariate.GammaUnivariate,
        'uniform': copulas.univariate.UniformUnivariate,
        'gaussian_kde': copulas.univariate.GaussianKDE,
    }

    @classmethod
    def get_distribution_class(cls, distribution):
        """Return the corresponding distribution class from ``copulas.univariate``.

        Args:
            distribution (str):
                A string representing a copulas univariate distribution.

        Returns:
            copulas.univariate:
                A copulas univariate class that corresponds to the distribution.
        """
        if not isinstance(distribution, str) or distribution not in cls._DISTRIBUTIONS:
            error_message = f"Invalid distribution specification '{distribution}'."
            raise ValueError(error_message)

        return cls._DISTRIBUTIONS[distribution]

    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=True,
                 locales=['en_US'], numerical_distributions=None, default_distribution=None):
        super().__init__(
            metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales,
        )
        validate_numerical_distributions(numerical_distributions, self.metadata.columns)
        self.numerical_distributions = numerical_distributions or {}
        self.default_distribution = default_distribution or 'beta'

        self._default_distribution = self.get_distribution_class(self.default_distribution)
        self._numerical_distributions = {
            field: self.get_distribution_class(distribution)
            for field, distribution in self.numerical_distributions.items()
        }
        self._num_rows = None

    def _fit(self, processed_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        log_numerical_distributions_error(
            self.numerical_distributions, processed_data.columns, LOGGER)
        self._num_rows = len(processed_data)

        numerical_distributions = deepcopy(self._numerical_distributions)
        for column in processed_data.columns:
            if column not in numerical_distributions:
                numerical_distributions[column] = self._numerical_distributions.get(
                    column, self._default_distribution)

        self._model = GaussianMultivariateMod(
            distribution=numerical_distributions
        )

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='scipy')
            self._model.fit(processed_data)

    def _warn_for_update_transformers(self, column_name_to_transformer):
        """Raise warnings for update_transformers.

        Args:
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.
        """
        for column, transformer in column_name_to_transformer.items():
            if isinstance(transformer, OneHotEncoder):
                warnings.warn(
                    f"Using a OneHotEncoder transformer for column '{column}' "
                    'may slow down the preprocessing and modeling times.'
                )

    def _sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates ``num_rows`` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        return self._model.sample(num_rows, conditions=conditions)

    def _get_valid_columns_from_metadata(self, columns):
        valid_columns = []
        for column in columns:
            for valid_column in self.metadata.columns:
                if column.startswith(valid_column):
                    valid_columns.append(column)
                    break

        return valid_columns

    def get_learned_distributions(self):
        """Get the marginal distributions used by the ``GaussianCopula``.

        Return a dictionary mapping the column names with the distribution name and the learned
        parameters for those.

        Returns:
            dict:
                Dictionary containing the distributions used or detected for each column and the
                learned parameters for those.
        """
        if not self._fitted:
            raise ValueError(
                "Distributions have not been learned yet. Please fit your model first using 'fit'."
            )

        parameters = self._model.to_dict()
        columns = parameters['columns']
        univariates = deepcopy(parameters['univariates'])
        learned_distributions = {}
        valid_columns = self._get_valid_columns_from_metadata(columns)
        for column, learned_params in zip(columns, univariates):
            if column in valid_columns:
                distribution = self.numerical_distributions.get(column, self.default_distribution)
                learned_params.pop('type')
                learned_distributions[column] = {
                    'distribution': distribution,
                    'learned_parameters': learned_params
                }

        return learned_distributions

    def _get_parameters(self):
        """Get copula model parameters.

        Compute model ``correlation`` and ``distribution.std``
        before it returns the flatten dict.

        Returns:
            dict:
                Copula parameters.

        Raises:
            NonParametricError:
                If a non-parametric distribution has been used.
        """
        for univariate in self._model.univariates:
            univariate_type = type(univariate)
            if univariate_type is copulas.univariate.Univariate:
                univariate = univariate._instance

            if univariate.PARAMETRIC == copulas.univariate.ParametricType.NON_PARAMETRIC:
                raise NonParametricError('This GaussianCopula uses non parametric distributions')

        params = self._model.to_dict()

        correlation = []
        for index, row in enumerate(params['correlation'][1:]):
            correlation.append(row[:index + 1])

        params['correlation'] = correlation
        params['univariates'] = dict(zip(params.pop('columns'), params['univariates']))
        params['num_rows'] = self._num_rows

        return flatten_dict(params)

    @staticmethod
    def _get_nearest_correlation_matrix(matrix):
        """Find the nearest correlation matrix.

        If the given matrix is not Positive Semi-definite, which means
        that any of its eigenvalues is negative, find the nearest PSD matrix
        by setting the negative eigenvalues to 0 and rebuilding the matrix
        from the same eigenvectors and the modified eigenvalues.

        After this, the matrix will be PSD but may not have 1s in the diagonal,
        so the diagonal is replaced by 1s and then the PSD condition of the
        matrix is validated again, repeating the process until the built matrix
        contains 1s in all the diagonal and is PSD.

        After 10 iterations, the last step is skipped and the current PSD matrix
        is returned even if it does not have all 1s in the diagonal.

        Insipired by: https://stackoverflow.com/a/63131250
        """
        eigenvalues, eigenvectors = scipy.linalg.eigh(matrix)
        negative = eigenvalues < 0
        identity = np.identity(len(matrix))

        iterations = 0
        while np.any(negative):
            eigenvalues[negative] = 0
            matrix = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T)
            if iterations >= 10:
                break

            matrix = matrix - matrix * identity + identity

            max_value = np.abs(np.abs(matrix).max())
            if max_value > 1:
                matrix /= max_value

            eigenvalues, eigenvectors = scipy.linalg.eigh(matrix)
            negative = eigenvalues < 0
            iterations += 1

        return matrix

    @classmethod
    def _rebuild_correlation_matrix(cls, triangular_correlation):
        """Rebuild a valid correlation matrix from its lower half triangle.

        The input of this function is a list of lists of floats of size 1, 2, 3...n-1:

           [[c_{2,1}], [c_{3,1}, c_{3,2}], ..., [c_{n,1},...,c_{n,n-1}]]

        Corresponding to the values from the lower half of the original correlation matrix,
        **excluding** the diagonal.

        The output is the complete correlation matrix reconstructed using the given values
        and scaled to the :math:`[-1, 1]` range if necessary.

        Args:
            triangle_correlation (list[list[float]]):
                A list that contains lists of floats of size 1, 2, 3... up to ``n-1``,
                where ``n`` is the size of the target correlation matrix.

        Returns:
            numpy.ndarray:
                rebuilt correlation matrix.
        """
        zero = [0.0]
        size = len(triangular_correlation) + 1
        left = np.zeros((size, size))
        right = np.zeros((size, size))
        for idx, values in enumerate(triangular_correlation):
            values = values + zero * (size - idx - 1)
            left[idx + 1, :] = values
            right[:, idx + 1] = values

        correlation = left + right
        max_value = np.abs(correlation).max()
        if max_value > 1:
            correlation /= max_value

        correlation += np.identity(size)

        return cls._get_nearest_correlation_matrix(correlation).tolist()

    def _rebuild_gaussian_copula(self, model_parameters, default_params=None):
        """Rebuild the model params to recreate a Gaussian Multivariate instance.

        Args:
            model_parameters (dict):
                Sampled and reestructured model parameters.
            default_parameters (dict):
                Fall back parameters if sampled params are invalid.

        Returns:
            dict:
                Model parameters ready to recreate the model.
        """
        if default_params is None:
            default_params = {}

        columns = []
        univariates = []
        for column, univariate in model_parameters['univariates'].items():
            columns.append(column)
            if column in self._numerical_distributions:
                univariate_type = self._numerical_distributions[column]
            else:
                univariate_type = self.get_distribution_class(self.default_distribution)

            univariate['type'] = univariate_type
            model = univariate_type.MODEL_CLASS
            if hasattr(model, '_argcheck'):
                to_check = {
                    parameter: univariate[parameter]
                    for parameter in inspect.signature(model._argcheck).parameters.keys()
                    if parameter in univariate
                }
                if not model._argcheck(**to_check):
                    if column in default_params.get('univariates', []):
                        LOGGER.info(
                            f"Invalid parameters sampled for column '{column}', "
                            'using default parameters.'
                        )
                        univariate = default_params['univariates'][column]
                        univariate['type'] = univariate_type
                    else:
                        LOGGER.debug(
                            f"Column '{column}' has invalid parameters."
                        )
            else:
                LOGGER.debug(f"Univariate for col '{column}' does not have _argcheck method.")

            if 'scale' in univariate:
                univariate['scale'] = max(0, univariate['scale'])

            univariates.append(univariate)

        model_parameters['univariates'] = univariates
        model_parameters['columns'] = columns

        correlation = model_parameters.get('correlation')
        if correlation:
            model_parameters['correlation'] = self._rebuild_correlation_matrix(correlation)
        else:
            model_parameters['correlation'] = [[1.0]]

        return model_parameters

    def _get_likelihood(self, table_rows):
        return self._model.probability_density(table_rows)

    def _set_parameters(self, parameters, default_params=None):
        """Set copula model parameters.

        Args:
            params [dict]:
                Copula flatten parameters.
            default_params [list]:
                Flattened list of parameters to fall back to if `params` are invalid.

        """
        if default_params is not None:
            default_params = unflatten_dict(default_params)
        else:
            default_params = {}

        parameters = unflatten_dict(parameters)
        if 'num_rows' in parameters:
            num_rows = parameters.pop('num_rows')
            self._num_rows = 0 if pd.isna(num_rows) else max(0, int(round(num_rows)))

        if parameters:
            parameters = self._rebuild_gaussian_copula(parameters, default_params)
            self._model = multivariate.GaussianMultivariate.from_dict(parameters)
