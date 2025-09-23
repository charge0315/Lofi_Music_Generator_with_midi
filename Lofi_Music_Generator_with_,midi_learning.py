# 🎵 Lofi Music Generator - 学習フェーズ
# MIDIデータで学習された音楽モデルからLofi音楽を生成します

#@title 1. 【環境設定】必要なライブラリのインストール
print("🎹 ライブラリのインストールを開始します...")
!pip install -q pretty_midi
!pip install -q tensorflow
!pip install -q music21
!pip install -q pydub
!pip install -q mido
print("✅ ライブラリのインストールが完了しました！")

#@title 2. 【ライブラリ読み込み】必要なモジュールをインポート
print("📚 必要なモジュールを読み込んでいます...")
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pretty_midi
import pickle
import glob
from typing import List, Tuple
import random
from IPython.display import Audio, display
import matplotlib.pyplot as plt
from google.colab import drive
import zipfile
import requests
from tqdm import tqdm
from pydub import AudioSegment
from pydub.generators import Sine
import warnings
warnings.filterwarnings('ignore')
print("✅ モジュールの読み込みが完了しました！")

#@title 3. 【Google Drive連携】データ保存用
print("💾 Google Driveをマウントしています...")
drive.mount('/content/drive')

# 作業ディレクトリの作成
work_dir = '/content/drive/MyDrive/LofiMusicGenerator'
os.makedirs(work_dir, exist_ok=True)
os.makedirs(f'{work_dir}/models', exist_ok=True)
os.makedirs(f'{work_dir}/generated_music', exist_ok=True)
os.makedirs(f'{work_dir}/datasets', exist_ok=True)
print(f"✅ 作業ディレクトリ: {work_dir}")

#@title 4. 【データセット準備】複数のデータセットをダウンロード
print("🎼 データセットのダウンロードを開始します...")

# データセット設定
datasets_config = {
    'maestro': {
        'name': 'MAESTROデータセット（クラシックピアノ）',
        'url': 'https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip',
        'use_all': False,  # False: 100件, True: 全件
        'max_files': 100
    },
    'lmd': {
        'name': 'Lakh MIDI Dataset（多ジャンル）',
        'url': 'http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz',
        'use_all': False,  # False: 100件, True: 全件
        'max_files': 100
    }
}

# どのデータセットを使用するか選択
print("\n📊 使用するデータセットを選択してください:")
print("  1: MAESTROのみ（クラシックピアノ）")
print("  2: LMDのみ（多ジャンル）")
print("  3: 両方（推奨）")
dataset_choice = input("選択 (1/2/3, デフォルト: 1): ").strip()

if dataset_choice == '2':
    active_datasets = ['lmd']
elif dataset_choice == '3':
    active_datasets = ['maestro', 'lmd']
else:
    active_datasets = ['maestro']

print(f"\n✅ 選択されたデータセット: {', '.join(active_datasets)}")

# 全件使用するか確認
use_all = input("全件使用しますか？ (y/n, デフォルト: n): ").strip().lower() == 'y'
if use_all:
    print("  ⚠️ 全件使用：処理に時間がかかります")
    for key in datasets_config:
        datasets_config[key]['use_all'] = True
        datasets_config[key]['max_files'] = None

all_midi_files = []

# MAESTROデータセット
if 'maestro' in active_datasets:
    print(f"\n📥 {datasets_config['maestro']['name']}を準備中...")
    maestro_path = f'{work_dir}/datasets/maestro'
    os.makedirs(maestro_path, exist_ok=True)
    
    # ダウンロード済みかチェック
    maestro_files = glob.glob(f"{maestro_path}/**/*.mid*", recursive=True)
    
    if len(maestro_files) == 0:
        print("  ダウンロード中...")
        response = requests.get(datasets_config['maestro']['url'], stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        zip_path = f'{maestro_path}/maestro.zip'
        with open(zip_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="MAESTRO") as pbar:
                for data in response.iter_content(chunk_size=8192):
                    file.write(data)
                    pbar.update(len(data))
        
        print("  解凍中...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(maestro_path)
        os.remove(zip_path)
        
        maestro_files = glob.glob(f"{maestro_path}/**/*.mid*", recursive=True)
    
    # ファイル数制限
    if not datasets_config['maestro']['use_all']:
        maestro_files = maestro_files[:datasets_config['maestro']['max_files']]
    
    all_midi_files.extend(maestro_files)
    print(f"  ✅ MAESTRO: {len(maestro_files)}個のMIDIファイル")

# Lakh MIDI Dataset
if 'lmd' in active_datasets:
    print(f"\n📥 {datasets_config['lmd']['name']}を準備中...")
    lmd_path = f'{work_dir}/datasets/lmd'
    os.makedirs(lmd_path, exist_ok=True)
    
    # ダウンロード済みかチェック
    lmd_files = glob.glob(f"{lmd_path}/**/*.mid*", recursive=True)
    
    if len(lmd_files) == 0:
        print("  ダウンロード中（約500MB）...")
        import tarfile
        
        response = requests.get(datasets_config['lmd']['url'], stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        tar_path = f'{lmd_path}/lmd.tar.gz'
        with open(tar_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="LMD") as pbar:
                for data in response.iter_content(chunk_size=8192):
                    file.write(data)
                    pbar.update(len(data))
        
        print("  解凍中...")
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(lmd_path)
        os.remove(tar_path)
        
        lmd_files = glob.glob(f"{lmd_path}/**/*.mid*", recursive=True)
    
    # ファイル数制限
    if not datasets_config['lmd']['use_all']:
        # ランダムに選択（多様性を確保）
        import random
        random.shuffle(lmd_files)
        lmd_files = lmd_files[:datasets_config['lmd']['max_files']]
    
    all_midi_files.extend(lmd_files)
    print(f"  ✅ LMD: {len(lmd_files)}個のMIDIファイル")

# 最終確認
print(f"\n✅ 合計 {len(all_midi_files)}個のMIDIファイルを処理対象にします")
midi_files = all_midi_files

#@title 5. 【データ前処理】MIDIファイルを統一フォーマットに変換
print("🔧 MIDIファイルの前処理を開始します...")

class MIDIProcessor:
    """MIDIファイルを処理して学習用データに変換"""
    
    def __init__(self, sequence_length: int = 256):
        self.sequence_length = sequence_length
        self.vocab_size = 388  # ピッチ(128) + ベロシティ(32) + 時間(100) + 特殊トークン(128)
        
    def midi_to_notes(self, midi_file: str) -> List[Tuple[int, int, float, float]]:
        """MIDIファイルをノートのリストに変換"""
        try:
            pm = pretty_midi.PrettyMIDI(midi_file)
            notes = []
            
            for instrument in pm.instruments:
                if not instrument.is_drum:  # ドラムトラックは除外
                    for note in instrument.notes:
                        notes.append((
                            note.pitch,
                            note.velocity,
                            note.start,
                            note.end - note.start  # duration
                        ))
            
            # 時間順にソート
            notes.sort(key=lambda x: x[2])
            return notes
        except Exception as e:
            return []
    
    def notes_to_sequence(self, notes: List[Tuple]) -> np.ndarray:
        """ノートリストを数値シーケンスに変換"""
        sequence = []
        
        for pitch, velocity, start, duration in notes:
            # トークン化
            pitch_token = min(pitch, 127)
            velocity_token = min(velocity // 4, 31) + 128
            time_token = min(int(start * 10), 99) + 160
            duration_token = min(int(duration * 10), 99) + 260
            
            sequence.extend([pitch_token, velocity_token, time_token, duration_token])
        
        # パディングまたはトリミング
        if len(sequence) < self.sequence_length:
            sequence.extend([0] * (self.sequence_length - len(sequence)))
        else:
            sequence = sequence[:self.sequence_length]
        
        return np.array(sequence, dtype=np.int32)

# プロセッサーのインスタンス化
processor = MIDIProcessor(sequence_length=256)

# データセットごとに処理
sequences_by_dataset = {}
total_sequences = []

# MAESTROの処理
maestro_sequences = []
if 'maestro' in active_datasets:
    maestro_files = [f for f in midi_files if 'maestro' in f]
    if maestro_files:
        print(f"\n  🎹 MAESTRO: {len(maestro_files)}個のファイルを処理中...")
        for midi_file in tqdm(maestro_files, desc="MAESTRO処理"):
            notes = processor.midi_to_notes(midi_file)
            if notes:
                sequence = processor.notes_to_sequence(notes)
                maestro_sequences.append(sequence)
        
        sequences_by_dataset['maestro'] = np.array(maestro_sequences)
        total_sequences.extend(maestro_sequences)
        print(f"    ✅ {len(maestro_sequences)}個のシーケンスを生成")

# LMDの処理
lmd_sequences = []
if 'lmd' in active_datasets:
    lmd_files = [f for f in midi_files if 'lmd' in f]
    if lmd_files:
        print(f"\n  🎵 LMD: {len(lmd_files)}個のファイルを処理中...")
        for midi_file in tqdm(lmd_files, desc="LMD処理"):
            notes = processor.midi_to_notes(midi_file)
            if notes and len(notes) > 10:  # 短すぎるファイルは除外
                sequence = processor.notes_to_sequence(notes)
                lmd_sequences.append(sequence)
        
        sequences_by_dataset['lmd'] = np.array(lmd_sequences)
        total_sequences.extend(lmd_sequences)
        print(f"    ✅ {len(lmd_sequences)}個のシーケンスを生成")

# 全データを結合
X_train = np.array(total_sequences) if total_sequences else np.random.randint(0, processor.vocab_size, size=(100, 256))

print(f"\n✅ 前処理完了！")
print(f"  総シーケンス数: {len(X_train)}")
if sequences_by_dataset:
    for name, seqs in sequences_by_dataset.items():
        print(f"  - {name}: {len(seqs)}個")

# データが少ない場合の警告
if len(X_train) < 50:
    print("\n  ⚠️ データが少ないため、学習結果が不安定になる可能性があります")
    print("     より多くのデータセットを使用することをお勧めします")

#@title 6. 【Music Transformer】モデル構築
print("🤖 Music Transformerモデルを構築しています...")

class MusicTransformer(keras.Model):
    """シンプルなMusic Transformerモデル"""
    
    def __init__(self, vocab_size: int, d_model: int = 256):
        super().__init__()
        
        # 埋め込み層
        self.embedding = layers.Embedding(vocab_size, d_model)
        
        # LSTM層（Transformerの簡易版として）
        self.lstm1 = layers.LSTM(d_model, return_sequences=True)
        self.lstm2 = layers.LSTM(d_model, return_sequences=True)
        self.dropout = layers.Dropout(0.2)
        
        # 出力層
        self.dense = layers.Dense(d_model, activation='relu')
        self.output_layer = layers.Dense(vocab_size, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.lstm1(x)
        x = self.dropout(x, training=training)
        x = self.lstm2(x)
        x = self.dropout(x, training=training)
        x = self.dense(x)
        return self.output_layer(x)

# モデルのインスタンス化
model = MusicTransformer(vocab_size=processor.vocab_size)

# モデルのコンパイル
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ モデルの構築が完了しました！")

#@title 7. 【学習実行】モデルの訓練
print("🎯 モデルの学習を開始します...")

# 学習データの準備
X = X_train[:, :-1]
y = X_train[:, 1:]

# 学習の実行（エポック数を少なめに設定）
print("  📈 学習中... (これには数分かかる場合があります)")
history = model.fit(
    X, y,
    batch_size=32,
    epochs=5,  # デモ用に少なめ
    validation_split=0.1,
    verbose=1
)

# 学習曲線の表示
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print("✅ モデルの学習が完了しました！")

#@title 8. 【音楽生成】Lofi音楽を生成
print("🎵 Lofi音楽を生成しています...")

def generate_music(model, processor, seed_sequence, length=512, temperature=0.8):
    """音楽シーケンスを生成"""
    generated = list(seed_sequence)
    
    for _ in tqdm(range(length), desc="生成中"):
        input_seq = np.array(generated[-255:]).reshape(1, -1)
        predictions = model.predict(input_seq, verbose=0)[0, -1, :]
        
        # temperature適用
        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))
        
        # サンプリング
        predicted_token = np.random.choice(len(predictions), p=predictions)
        generated.append(predicted_token)
    
    return np.array(generated)

def sequence_to_midi(sequence, output_file, tempo=75):
    """シーケンスをMIDIファイルに変換"""
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=4)  # Electric Piano
    
    current_time = 0.0
    
    for i in range(0, len(sequence) - 3, 4):
        pitch = min(sequence[i], 127)
        velocity = min((sequence[i + 1] - 128) * 4, 127) if sequence[i + 1] >= 128 else 64
        start_time = (sequence[i + 2] - 160) * 0.1 if sequence[i + 2] >= 160 else current_time
        duration = (sequence[i + 3] - 260) * 0.1 if sequence[i + 3] >= 260 else 0.5
        
        if pitch > 0 and velocity > 0:
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=start_time + duration
            )
            piano.notes.append(note)
            current_time = start_time + duration
    
    pm.instruments.append(piano)
    pm.write(output_file)
    return pm

# 3つのトラックを生成
print("  🎹 3つのLofiトラックを生成します...")

generated_files = []
for i in range(3):
    print(f"\n  🎵 トラック {i+1} を生成中...")
    
    # シードシーケンス
    if len(X_train) > 0:
        seed_idx = np.random.randint(0, len(X_train))
        seed_sequence = X_train[seed_idx, :128]
    else:
        seed_sequence = np.random.randint(0, processor.vocab_size, size=128)
    
    # 生成
    temperature = 0.6 + i * 0.2
    generated_sequence = generate_music(model, processor, seed_sequence, length=256, temperature=temperature)
    
    # MIDI保存
    midi_file = f"{work_dir}/generated_music/lofi_track_{i+1}.mid"
    pm = sequence_to_midi(generated_sequence, midi_file, tempo=70 + i * 5)
    generated_files.append(midi_file)
    
    print(f"  ✅ トラック {i+1} 生成完了！")

print("\n✅ Lofi音楽の生成が完了しました！")

#@title 9. 【モデル保存】学習済モデルをGoogle Driveに保存
print("💾 モデルを保存しています...")

# モデルの保存
model_path = f"{work_dir}/models/lofi_music_transformer.h5"
model.save(model_path)
print(f"  ✅ モデル保存完了: {model_path}")

# プロセッサーの設定も保存
processor_config = {
    'sequence_length': processor.sequence_length,
    'vocab_size': processor.vocab_size
}

config_path = f"{work_dir}/models/processor_config.pkl"
with open(config_path, 'wb') as f:
    pickle.dump(processor_config, f)
print(f"  ✅ 設定保存完了: {config_path}")

#@title 10. 【MP3保存】生成した音楽をMP3形式で保存
print("🎧 MP3ファイルを生成しています...")

def midi_to_mp3(midi_file, mp3_file, duration=30):
    """MIDIをシンプルなMP3に変換（簡易シンセサイザー）"""
    try:
        pm = pretty_midi.PrettyMIDI(midi_file)
        
        # オーディオを生成（簡易版）
        sample_rate = 44100
        audio_length = int(min(duration, pm.get_end_time()) * sample_rate)
        audio = np.zeros(audio_length)
        
        for instrument in pm.instruments:
            for note in instrument.notes[:50]:  # 最初の50ノートのみ
                start_sample = int(note.start * sample_rate)
                end_sample = min(int(note.end * sample_rate), audio_length)
                
                if start_sample < audio_length:
                    # 周波数計算
                    frequency = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
                    
                    # サイン波生成
                    t = np.arange(end_sample - start_sample) / sample_rate
                    wave = np.sin(2 * np.pi * frequency * t) * (note.velocity / 127.0) * 0.3
                    
                    # オーディオに追加
                    end_idx = min(start_sample + len(wave), audio_length)
                    audio[start_sample:end_idx] += wave[:end_idx - start_sample]
        
        # 正規化
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        # pydubで変換
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        
        # MP3として保存
        audio_segment.export(mp3_file, format="mp3", bitrate="192k")
        return True
        
    except Exception as e:
        print(f"  ⚠️ MP3変換エラー: {e}")
        return False

# 生成したMIDIファイルをMP3に変換
mp3_files = []
for i, midi_file in enumerate(generated_files, 1):
    mp3_file = midi_file.replace('.mid', '.mp3')
    print(f"  🎵 トラック {i} をMP3に変換中...")
    
    if midi_to_mp3(midi_file, mp3_file, duration=30):
        mp3_files.append(mp3_file)
        print(f"  ✅ MP3生成完了: {os.path.basename(mp3_file)}")
    else:
        print(f"  ⚠️ MP3変換をスキップ")

print(f"\n✅ {len(mp3_files)}個のMP3ファイルを生成しました！")

# 完了メッセージ
print("\n" + "="*60)
print("🎉 学習フェーズが完了しました！")
print("="*60)
print(f"\n📁 生成されたファイル:")
print(f"  モデル: {work_dir}/models/")
print(f"  音楽: {work_dir}/generated_music/")
print("\n💡 次は音楽再生フェーズを実行してください！")