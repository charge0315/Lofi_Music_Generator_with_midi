# 🎛️ Lofi Music Custom Generator - カスタム生成フェーズ
# カスタマイズしたLofi音楽を生成します

#@title 1. 【カスタマイズ】Lofi音楽のスタイル調整
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pretty_midi
from google.colab import drive
import pickle
from IPython.display import Audio, display
from datetime import datetime
from pydub import AudioSegment
import warnings
warnings.filterwarnings('ignore')

print("🎛️ カスタム生成ツールを起動中...")

# Google Driveをマウント
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

work_dir = '/content/drive/MyDrive/LofiMusicGenerator'

# モデルとプロセッサー設定を読み込み
try:
    print("  📥 学習済みモデルを読み込み中...")
    model_path = f"{work_dir}/models/lofi_music_transformer.h5"
    model = keras.models.load_model(model_path)
    
    with open(f"{work_dir}/models/processor_config.pkl", 'rb') as f:
        processor_config = pickle.load(f)
    
    print("  ✅ モデルの読み込み完了！")
    model_loaded = True
    
except Exception as e:
    print(f"  ⚠️ モデル読み込みエラー: {e}")
    print("  先に学習フェーズを実行してください")
    model_loaded = False
    model = None
    processor_config = {'vocab_size': 388, 'sequence_length': 256}

# スタイル設定
print("\n🎨 利用可能なLofiスタイル:")
print("="*60)

styles = {
    'chill': {
        'name': '🌙 Chill（リラックス）',
        'tempo': 70,
        'temperature': 0.7,
        'pitch_shift': 0
    },
    'jazzy': {
        'name': '🎷 Jazzy（ジャズ風）',
        'tempo': 85,
        'temperature': 0.9,
        'pitch_shift': 2
    },
    'study': {
        'name': '📚 Study（勉強用）',
        'tempo': 75,
        'temperature': 0.6,
        'pitch_shift': -2
    },
    'nostalgic': {
        'name': '📼 Nostalgic（懐かしい）',
        'tempo': 72,
        'temperature': 0.8,
        'pitch_shift': -3
    },
    'ambient': {
        'name': '🌊 Ambient（環境音楽）',
        'tempo': 60,
        'temperature': 1.0,
        'pitch_shift': 0
    }
}

# スタイル一覧を表示
for key, style in styles.items():
    print(f"  [{key}] {style['name']}")
    print(f"      テンポ: {style['tempo']} BPM, 複雑さ: {style['temperature']:.1f}")

print("="*60)

# カスタム生成関数
def generate_custom_music(style_key='chill', track_name=None, length=256):
    """カスタマイズされたLofi音楽を生成"""
    
    if not model_loaded:
        print("⚠️ モデルが読み込まれていません")
        return None
    
    # スタイル選択
    if style_key not in styles:
        style_key = 'chill'
    style = styles[style_key]
    
    # トラック名
    if not track_name:
        timestamp = datetime.now().strftime("%H%M%S")
        track_name = f"custom_{style_key}_{timestamp}"
    
    print(f"\n🎵 「{track_name}」を生成中...")
    print(f"  スタイル: {style['name']}")
    print(f"  長さ: {length}ステップ")
    print("-"*40)
    
    # シードシーケンス生成
    seed = np.random.randint(0, processor_config['vocab_size'], size=128)
    
    # 音楽生成
    generated = list(seed)
    
    print("  生成中...")
    for i in range(length):
        if i % 50 == 0 and i > 0:
            print(f"  進捗: {i}/{length}")
        
        input_seq = np.array(generated[-255:]).reshape(1, -1)
        predictions = model.predict(input_seq, verbose=0)[0, -1, :]
        
        # temperature適用
        predictions = np.log(predictions + 1e-10) / style['temperature']
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))
        
        predicted_token = np.random.choice(len(predictions), p=predictions)
        generated.append(predicted_token)
    
    print("  ✅ 生成完了！")
    
    # MIDIファイルに変換
    midi_file = f"{work_dir}/generated_music/{track_name}.mid"
    pm = pretty_midi.PrettyMIDI(initial_tempo=style['tempo'])
    piano = pretty_midi.Instrument(program=4)  # Electric Piano
    
    current_time = 0.0
    note_count = 0
    
    for i in range(0, len(generated) - 3, 4):
        pitch = min(generated[i], 127)
        pitch += style['pitch_shift']  # スタイルに応じてピッチシフト
        pitch = max(0, min(127, pitch))
        
        velocity = min((generated[i + 1] - 128) * 4, 127) if generated[i + 1] >= 128 else 64
        start_time = (generated[i + 2] - 160) * 0.1 if generated[i + 2] >= 160 else current_time
        duration = (generated[i + 3] - 260) * 0.1 if generated[i + 3] >= 260 else 0.5
        
        if pitch > 0 and velocity > 0:
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=start_time + duration
            )
            piano.notes.append(note)
            current_time = start_time + duration
            note_count += 1
    
    pm.instruments.append(piano)
    pm.write(midi_file)
    
    print(f"\n📊 生成情報:")
    print(f"  ファイル: {track_name}.mid")
    print(f"  ノート数: {note_count}")
    print(f"  長さ: {pm.get_end_time():.1f}秒")
    
    # MP3に変換
    mp3_file = midi_file.replace('.mid', '.mp3')
    if convert_to_mp3(midi_file, mp3_file):
        print(f"  ✅ MP3変換完了: {track_name}.mp3")
        return mp3_file
    
    return midi_file

def convert_to_mp3(midi_file, mp3_file):
    """MIDIをMP3に変換"""
    try:
        pm = pretty_midi.PrettyMIDI(midi_file)
        
        # 簡易シンセサイザー
        sample_rate = 22050
        duration = min(30, pm.get_end_time())
        audio_length = int(duration * sample_rate)
        audio = np.zeros(audio_length)
        
        for instrument in pm.instruments:
            for note in instrument.notes[:50]:
                start = int(note.start * sample_rate)
                end = min(int(note.end * sample_rate), audio_length)
                
                if start < audio_length:
                    freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
                    t = np.arange(end - start) / sample_rate
                    wave = np.sin(2 * np.pi * freq * t) * 0.3
                    
                    # Lofi効果（ビットクラッシュ）
                    wave = np.round(wave * 8) / 8
                    
                    end_idx = min(start + len(wave), audio_length)
                    audio[start:end_idx] += wave[:end_idx - start]
        
        # 正規化
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        # MP3保存
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        audio_segment.export(mp3_file, format="mp3", bitrate="192k")
        
        return True
        
    except Exception as e:
        print(f"  ⚠️ MP3変換エラー: {e}")
        return False

#@title 2. 【使用例】Lofi音楽を生成して再生

def quick_generate():
    """クイック生成（デフォルト設定）"""
    if model_loaded:
        print("\n⚡ クイック生成を開始...")
        file = generate_custom_music(style_key='chill', length=256)
        
        if file and file.endswith('.mp3'):
            print("\n🎧 生成した音楽を再生:")
            display(Audio(file, autoplay=True))
        
        return file
    else:
        print("⚠️ モデルが読み込まれていません")
        return None

def custom_generate():
    """カスタム生成（対話的）"""
    if not model_loaded:
        print("⚠️ モデルが読み込まれていません")
        return None
    
    print("\n🎛️ カスタム生成設定")
    print("="*60)
    
    # スタイル選択
    print("\n利用可能なスタイル:")
    for key in styles:
        print(f"  - {key}")
    
    style_key = input("\nスタイルを選択 (デフォルト: chill): ").strip().lower()
    if style_key not in styles:
        style_key = 'chill'
    
    # トラック名
    track_name = input("トラック名 (省略可): ").strip()
    if not track_name:
        track_name = None
    
    # 長さ選択
    print("\n長さ:")
    print("  1: ショート (128ステップ)")
    print("  2: ミディアム (256ステップ)")
    print("  3: ロング (512ステップ)")
    
    length_choice = input("選択 (1/2/3, デフォルト: 2): ").strip()
    length_map = {'1': 128, '2': 256, '3': 512}
    length = length_map.get(length_choice, 256)
    
    # 生成実行
    file = generate_custom_music(style_key=style_key, track_name=track_name, length=length)
    
    if file and file.endswith('.mp3'):
        print("\n🎧 生成した音楽を再生:")
        display(Audio(file, autoplay=True))
    
    return file

def batch_generate():
    """バッチ生成（複数スタイルで一度に生成）"""
    if not model_loaded:
        print("⚠️ モデルが読み込まれていません")
        return []
    
    print("\n📦 バッチ生成を開始...")
    print("  各スタイルで1曲ずつ生成します")
    print("="*60)
    
    generated_files = []
    
    for style_key in ['chill', 'jazzy', 'study']:
        print(f"\n[{style_key}] {styles[style_key]['name']}を生成中...")
        
        track_name = f"batch_{style_key}_{datetime.now().strftime('%H%M%S')}"
        file = generate_custom_music(style_key=style_key, track_name=track_name, length=256)
        
        if file:
            generated_files.append(file)
    
    print("\n✅ バッチ生成完了！")
    print(f"  生成されたファイル: {len(generated_files)}個")
    
    return generated_files

# メインメニュー
if model_loaded:
    print("\n" + "="*60)
    print("🎮 操作方法:")
    print("="*60)
    print("  quick_generate()   # クイック生成（おすすめ）")
    print("  custom_generate()  # カスタム生成")
    print("  batch_generate()   # バッチ生成")
    print("="*60)
    
    print("\n💡 まずは quick_generate() を実行してみてください！")
    
else:
    print("\n⚠️ 学習済みモデルが見つかりません")
    print("  学習フェーズを先に実行してください")

# デモ実行
def run_demo():
    """簡単なデモを実行"""
    if model_loaded:
        print("\n🎉 デモを実行します！")
        print("="*60)
        
        # Chillスタイルで生成
        print("\n1. Chillスタイルで生成...")
        file1 = generate_custom_music('chill', 'demo_chill', 128)
        
        # Jazzyスタイルで生成
        print("\n2. Jazzyスタイルで生成...")
        file2 = generate_custom_music('jazzy', 'demo_jazzy', 128)
        
        print("\n✅ デモ完了！")
        
        # 再生
        if file1 and file1.endswith('.mp3'):
            print("\n🎧 Chill:")
            display(Audio(file1))
        
        if file2 and file2.endswith('.mp3'):
            print("\n🎧 Jazzy:")
            display(Audio(file2))
        
        print("\n🎉 Lofi音楽生成デモを完了しました！")
        
    else:
        print("⚠️ モデルが読み込まれていません")