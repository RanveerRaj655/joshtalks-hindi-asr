import requests
import json
import os

def print_schema(data, indent=0):
    """
    Recursively prints the schema (keys and types) of a JSON object/list.
    """
    prefix = "  " * indent
    if isinstance(data, dict):
        print(f"{prefix}{{")
        for key, value in data.items():
            type_name = type(value).__name__
            if isinstance(value, (dict, list)):
                print(f"{prefix}  \"{key}\": ", end="")
                print_schema(value, indent + 1)
            else:
                print(f"{prefix}  \"{key}\": {type_name}")
        print(f"{prefix}}}")
    elif isinstance(data, list):
        if len(data) > 0:
            print(f"{prefix}[")
            print_schema(data[0], indent + 1)
            print(f"{prefix}  ... ({len(data)} items total)")
            print(f"{prefix}]")
        else:
            print(f"{prefix}[]")
    else:
        print(f"{prefix}{type(data).__name__}")

def inspect_asr_record(transcription_url):
    """
    Fetches transcription, audio, and metadata for a given Hindi ASR record.
    """
    print(f"\n{'='*60}")
    print(f"HINDI ASR DATASET INSPECTOR")
    print(f"{'='*60}")
    print(f"Target Record: {os.path.basename(transcription_url)}")
    print(f"URL: {transcription_url}\n")

    # 1. Fetch Transcription JSON
    try:
        response = requests.get(transcription_url)
        response.raise_for_status()
        transcription_data = response.json()
        print("1. [TRANSCRIPTION SCHEMA]")
        print_schema(transcription_data)
        print("\n[SAMPLE SEGMENT]:")
        if isinstance(transcription_data, list) and len(transcription_data) > 0:
            print(json.dumps(transcription_data[0], indent=4, ensure_ascii=False))
        else:
            print(json.dumps(transcription_data, indent=4, ensure_ascii=False))
    except Exception as e:
        print(f"Error fetching transcription: {e}")
        return

    print(f"\n{'-'*60}")

    # 2. Derive Audio URL
    # Pattern: replace _transcription.json with .wav or .mp3
    base_url = transcription_url.replace("_transcription.json", "")
    audio_formats = [".wav", ".mp3"]
    found_audio = None
    
    print("2. [AUDIO FILE SEARCH]")
    for ext in audio_formats:
        audio_url = f"{base_url}{ext}"
        try:
            head_resp = requests.head(audio_url, allow_redirects=True)
            if head_resp.status_code == 200:
                print(f"STATUS: FOUND")
                print(f"FORMAT: {ext}")
                print(f"URL   : {audio_url}")
                found_audio = audio_url
                break
        except Exception as e:
            continue
    
    if not found_audio:
        print("STATUS: NOT FOUND (Checked .wav and .mp3)")
        print(f"Note: Checked base name '{base_url}'")

    print(f"\n{'-'*60}")

    # 3. Fetch Metadata JSON
    metadata_url = transcription_url.replace("_transcription.json", "_metadata.json")
    print(f"3. [METADATA SCHEMA]")
    try:
        response = requests.get(metadata_url)
        if response.status_code == 200:
            metadata_data = response.json()
            print_schema(metadata_data)
            print("\n[FULL METADATA]:")
            print(json.dumps(metadata_data, indent=4, ensure_ascii=False))
        else:
            print(f"STATUS: NOT FOUND ({response.status_code})")
            print(f"URL   : {metadata_url}")
    except Exception as e:
        print(f"Error fetching metadata: {e}")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    # Example URL provided by the user
    example_url = "https://storage.googleapis.com/upload_goai/967179/825780_transcription.json"
    inspect_asr_record(example_url)
