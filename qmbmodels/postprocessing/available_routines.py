from .entropy import entro_ave, footer_entro, entro_post_analysis
from .entropy import footer_entro_analyse, \
    footer_entro_no_sample_averaging
from .rstat import footer_r_ave, r_ave

_routines_dict = {
    'get_entro_ave': [entro_ave, 'Entropy', 17, footer_entro],
    'get_entro_ave_samples': [entro_ave, 'Entropy', None,
                              footer_entro_no_sample_averaging],
    'entro_analyse': [entro_post_analysis, 'Entropy',
                      None, footer_entro_analyse],
    'get_r': [r_ave, 'r_data', 15, footer_r_ave],

}
