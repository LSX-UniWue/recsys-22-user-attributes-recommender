from recbole.quick_start import run_recbole

if __name__ == '__main__':
    run_recbole(model='BERT4Rec', dataset='ml-20m',
                config_dict={
                    'device': 'cpu',
                    'load_col': {
                        'inter': ['user_id', 'item_id', 'timestamp']
                    },
                    'eval_setting': 'RO_LS,uni100'
                })
