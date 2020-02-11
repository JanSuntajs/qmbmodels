from .entropy import entro_ave, footer_entro, entro_post_analysis
from .entropy import footer_entro_analyse
from .rstat import footer_r_ave, r_ave

_routines_dict = {
    'get_entro_ave': [entro_ave, 'Entropy', 17, footer_entro],
    'entro_analyse': [entro_post_analysis, 'Entropy',
                      None, footer_entro_analyse],
    'get_r': [r_ave, 'r_data', 15, footer_r_ave],

}
