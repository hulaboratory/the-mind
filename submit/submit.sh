#!/bin/bash -l
#SBATCH --job-name=32b-m
#SBATCH --partition=h100
#SBATCH --account=jhu35
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/"%x.o%j"
#SBATCH --error=logs/"%x.e%j"
#SBATCH --exclude=gh113
module load gcc/9.3.0
module load anaconda3/2024.02-1
conda activate mind

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "CUDA_HOME=$CUDA_HOME"

nvidia-smi

#python agent_play_llm_llm2.py gpt-oss-120b Qwen/Qwen3-32B time_based
#python agent_play_llm_llm2.py gpt-oss-120b Qwen/Qwen3-32B wait_count_based

#python agent_play_llm_llm.py gpt-oss-120b qwen3-32b time_based
#python agent_play_llm_llm.py gpt-oss-120b qwen3-32b wait_count_based
#python agent_play_llm_llm.py gpt-oss-120b qwen3-32b wait_based

#
#python agent_play_realtime.py gpt-oss-120b
#python agent_play_realtime.py gpt-oss-20b
#python agent_play_realtime_llm_llm.py Qwen/Qwen3-32B gemini
#python agent_play_realtime_llm_llm.py gpt-oss-120b gemini
#python agent_play_realtime_llm_llm.py gpt-oss-120b Qwen/Qwen3-32B
#python agent_play_llm_llm2.py Qwen/Qwen3-32B gemini wait_count_based
#python agent_play_realtime_gptoss.py gpt-oss-20b
#python agent_play_realtime_gptoss.py Qwen/Qwen3-32B
#python agent_play_llm_llm.py gemini gpt-oss-120b time_based
python agent_play_hand.py qwen3-32b time_based_millisecond 1 6
#python agent_play_llm_llm.py gemini gpt-oss-120b wait_count_based
#python agent_play.py gpt-oss-20b wait_count_based
#python agent_play_qwen4.py Qwen/Qwen3-8B time_based
#python agent_play_qwen3.py Qwen/Qwen3-8B time_based
#python agent_play_realtime_gptoss.py Qwen/Qwen3-32B
