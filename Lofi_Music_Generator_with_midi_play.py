# 🎧 Lofi Music Player - 音楽再生フェーズ
# 生成されたLofi音楽を再生します

#@title 1. 【MP3選択】MP3のリストを表示
import os
import glob
from google.colab import drive
from IPython.display import Audio, display
import numpy as np

print("🎵 Lofi Music Playerを起動しています...")

# Google Driveをマウント
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')
    print("✅ Google Driveをマウントしました")

# 作業ディレクトリ
work_dir = '/content/drive/MyDrive/LofiMusicGenerator'
music_dir = f'{work_dir}/generated_music'

# MP3ファイルのリストを取得
mp3_files = glob.glob(f"{music_dir}/*.mp3")
mp3_files.sort()

# MIDIファイルも確認
midi_files = glob.glob(f"{music_dir}/*.mid")
midi_files.sort()

print("\n" + "="*60)
print("📋 音楽ファイルリスト")
print("="*60)

if mp3_files:
    print("\n🎵 MP3ファイル:")
    for i, file in enumerate(mp3_files, 1):
        filename = os.path.basename(file)
        file_size = os.path.getsize(file) / 1024  # KB
        print(f"  [{i}] {filename} ({file_size:.1f} KB)")
else:
    print("\n⚠️ MP3ファイルが見つかりません")

if midi_files:
    print(f"\n🎹 MIDIファイル: {len(midi_files)}個")
    for file in midi_files[:5]:  # 最初の5個だけ表示
        print(f"  - {os.path.basename(file)}")
else:
    print("\n⚠️ MIDIファイルが見つかりません")

print("="*60)

#@title 2. 【音楽再生】選択した番号の音楽を試聴

def play_music_by_number(number: int):
    """指定された番号の音楽を再生"""
    if mp3_files and 0 < number <= len(mp3_files):
        mp3_file = mp3_files[number - 1]
        filename = os.path.basename(mp3_file)
        
        print(f"\n🎧 再生中: {filename}")
        print("="*60)
        
        # ファイル情報
        file_size = os.path.getsize(mp3_file) / 1024
        print(f"  📊 ファイルサイズ: {file_size:.1f} KB")
        print(f"  📁 パス: {mp3_file}")
        print("="*60)
        
        # 再生
        display(Audio(mp3_file, autoplay=True))
        print("\n✅ 上のプレイヤーで音楽をお楽しみください！")
        
    else:
        print(f"⚠️ 無効な番号です。1〜{len(mp3_files)}の番号を入力してください。")

# MP3が存在しない場合、簡易的にMIDIから音声を生成
def play_midi_simple(midi_file):
    """MIDIファイルを簡易的に再生"""
    try:
        import pretty_midi
        
        print(f"  🎹 MIDIファイルから音声を生成中...")
        pm = pretty_midi.PrettyMIDI(midi_file)
        
        # 簡易シンセサイザー
        sample_rate = 22050
        duration = min(30, pm.get_end_time())  # 最大30秒
        audio_length = int(duration * sample_rate)
        audio = np.zeros(audio_length)
        
        for instrument in pm.instruments:
            for note in instrument.notes[:30]:  # 最初の30ノート
                start = int(note.start * sample_rate)
                end = min(int(note.end * sample_rate), audio_length)
                
                if start < audio_length:
                    freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
                    t = np.arange(end - start) / sample_rate
                    wave = np.sin(2 * np.pi * freq * t) * 0.3
                    
                    end_idx = min(start + len(wave), audio_length)
                    audio[start:end_idx] += wave[:end_idx - start]
        
        # 正規化
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.5
        
        display(Audio(audio, rate=sample_rate, autoplay=True))
        print("  ✅ MIDI再生中...")
        
    except Exception as e:
        print(f"  ⚠️ MIDI再生エラー: {e}")

# 再生実行
if mp3_files:
    print("\n🎮 操作方法:")
    print("  下のセルで曲番号（1〜{}）を入力してください".format(len(mp3_files)))
    print("\n💡 例: play_music_by_number(1)")
    
elif midi_files:
    print("\n⚠️ MP3ファイルがないため、MIDIファイルを再生します")
    print("\n🎮 最初のMIDIファイルを再生中...")
    play_midi_simple(midi_files[0])
    
else:
    print("\n⚠️ 音楽ファイルが見つかりません")
    print("  先に学習フェーズを実行してください")

# インタラクティブな再生関数
def play_interactive():
    """対話的に曲を選択して再生"""
    if mp3_files:
        print("\n🎵 再生する曲を選択してください")
        print("="*60)
        
        for i, file in enumerate(mp3_files, 1):
            print(f"  [{i}] {os.path.basename(file)}")
        
        print("="*60)
        
        try:
            choice = int(input("曲番号を入力 (1〜{}): ".format(len(mp3_files))))
            play_music_by_number(choice)
        except ValueError:
            print("⚠️ 有効な番号を入力してください")
    else:
        print("⚠️ MP3ファイルがありません")

# プレイリスト再生
def play_all():
    """すべての曲を順番に再生"""
    if mp3_files:
        print(f"\n🎵 {len(mp3_files)}曲のプレイリストを再生します")
        print("="*60)
        
        for i, mp3_file in enumerate(mp3_files, 1):
            print(f"\n[{i}/{len(mp3_files)}] {os.path.basename(mp3_file)}")
            display(Audio(mp3_file))
        
        print("\n✅ プレイリスト準備完了！")
        print("  各プレイヤーで再生してください")
    else:
        print("⚠️ MP3ファイルがありません")

# 使用例を表示
print("\n" + "="*60)
print("📖 使用例:")
print("="*60)
print("  play_music_by_number(1)  # 1番の曲を再生")
print("  play_interactive()       # 対話的に選択")
print("  play_all()              # すべて再生")
print("="*60)