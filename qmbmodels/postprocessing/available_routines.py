from .entropy import entro_ave, footer_entro


_routines_dict = {
    #'get_entro_ave_post': [_entro_ave_postprocessed, 'Entropy', 13,
    #                      footer_entro_post],
    'get_entro_ave': [entro_ave, 'Entropy', 15, footer_entro],
    #'get_r': [_get_r, 'r_data', 6, footer_r]

}
