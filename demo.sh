set -e

# for how the demo code is set up, to run the "nosplit" variants of survival kernets with TUNA, you must run `python demo_xgb.py cfg_baselines.ini` first:
python demo_xgb.py cfg_baselines.ini
python demo_tuna_kernet.py cfg_kernet_nosplit_sft.ini  # with summary fine-tuning
# python demo_tuna_kernet.py cfg_kernet_nosplit.ini  # without summary fine-tuning

# for how the demo code is set up, to run the "split" variants of survival kernets with TUNA, you must run `python demo_xgb.py cfg_baselines_split.ini` first:
# python demo_xgb.py cfg_baselines_split.ini
# python demo_tuna_kernet.py cfg_kernet_split_sft.ini  # with summary fine-tuning
# python demo_tuna_kernet.py cfg_kernet_split.ini  # without summary fine-tuning

# for training survival kernets without TUNA, there is no need to train an XGBoost model first:
# python demo_kernet.py cfg_kernet_nosplit_sft.ini
# python demo_kernet.py cfg_kernet_nosplit.ini
# python demo_kernet.py cfg_kernet_split_sft.ini
# python demo_kernet.py cfg_kernet_split.ini
