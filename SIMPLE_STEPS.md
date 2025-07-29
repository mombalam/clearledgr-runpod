# Step-by-Step RunPod Setup for Clearledgr (Simple Version)

## Step 1: Create Your RunPod Pod ✅ (You've done this!)

## Step 2: Access Your Pod

1. Click "Connect" on your pod
2. Choose "Start Web Terminal" or "Connect via SSH"
3. You should see a terminal like this: `root@your-pod-id:/workspace#`

## Step 3: Download This Project

In your RunPod terminal, run:
```bash
cd /workspace
git clone https://github.com/your-username/clearledgr-2.0.git
cd clearledgr-2.0/runpod_setup
```

*(Replace `your-username` with your actual GitHub username)*

## Step 4: Get Your Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it "RunPod Training"
4. Select "Read" permission
5. Copy the token (starts with `hf_...`)

## Step 5: Set Your Token

In your RunPod terminal:
```bash
export HF_TOKEN=hf_your_token_here
```
*(Replace with your actual token)*

## Step 6: Request Llama Access

1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B
2. Click "Request access"
3. Fill out the form
4. Wait for approval (usually 1-2 hours)

## Step 7: Test Your Setup

```bash
python3 -c "print('Python is working!')"
nvidia-smi
```

You should see your GPU info displayed.

## What's Next?

Once you complete these steps, we'll continue with:
- Installing the required packages
- Downloading the Llama model
- Starting the training

**Let me know when you've completed these steps and I'll give you the next ones!** 👍
