from .entropy import entro_ave, footer_entro, entro_post_analysis
from .entropy import footer_entro_analyse, \
    footer_entro_no_sample_averaging
from .rstat import footer_r_ave, r_ave
from .sff import get_sff, footer_sff, get_tau_thouless, footer_t_thouless
from .degeneracies import deg_ave, footer_deg_ave
from .gamma import gamma_ave, footer_gamma_ave
from .microcan_deltae import microcan_ave, footer_microcan_ave
from .ipr import ipr_ave, footer_ipr_ave
from .thouless_conductivity import thoucond_ave, footer_thoucond_ave
from .kohn_conductivity import kohn_ave, footer_kohn_ave
# Routines available for calculations:
# get_entro_ave: average entropy, sampling averaging performed
#
# get_entro_ave_samples: no sampling averaging perfomed, results
# returned are given for each disorder realization
#
# entro_analyse: analysis w.r.t. to the number of samples
# included where the samples are ordered according to the
# difference of their variance w.r.t. to the theoretic
# prediction for the variance
#
# get_r: r calculations, sampling averaging performed.
_routines_dict = {
    'get_entro_ave': [entro_ave, 'Entropy', 17, footer_entro],
    'get_entro_ave_samples': [entro_ave, 'Entropy', None,
                              footer_entro_no_sample_averaging],
    'entro_analyse': [entro_post_analysis, 'Entropy',
                      None, footer_entro_analyse],
    'get_r': [r_ave, 'r_data', 15, footer_r_ave],
    'get_deg': [deg_ave, 'degeneracies_data', 14, footer_deg_ave],
    'get_gamma': [gamma_ave, 'gamma_data', 19, footer_gamma_ave],
    'get_microcan_deltae': [microcan_ave, 'deltaE_data', 19,
                            footer_microcan_ave],
    'get_sff': [get_sff, 'SFF_spectrum', None, footer_sff],
    'get_tau_thouless': [get_tau_thouless, 'SFF_spectrum', 25,
                         footer_t_thouless],
    'get_ipr': [ipr_ave, 'Ipr', 17, footer_ipr_ave],
    'get_thoucond': [thoucond_ave, 'Spectrum_differences', 23,
                     footer_thoucond_ave],
    'get_kohncond': [kohn_ave, 'kohn_data', 28,
                     footer_kohn_ave]

}
