This repository is based on:
https://github.com/AI-secure/semantic-randomized-smoothing
https://github.com/microsoft/denoised-smoothing
https://github.com/Jayfeather1024/DensePure
https://github.com/Ping-C/CertifiedObjectDetection


Run RS and DRS on adv images: 
cd ccrf
python3 calc_rs_{metric_name}.py

Run RS and DRS on clear images: 
cd ccrf
python3 calc_clear_rs_{metric_name}.py



Run MS and DMS on adv images: 
cd ccrf
python3 calc_nms_{metric_name}.py

Run MS and DMS on clear images: 
cd ccrf
python3 calc_clear_nms_{metric_name}.py



Run DDRS on adv images: 
cd ccrf
python3 calc_ddrs_{metric_name}.py

Run DDRS on clear images: 
cd ccrf
python3 calc_clear_ddrs_{metric_name}.py



Run DP on adv images: 
cd DensePure-master
python3 calc_dp_{metric_name}.py

Run DP on clear images: 
cd DensePure-master
python3 calc_clear_dp_{metric_name}.py
