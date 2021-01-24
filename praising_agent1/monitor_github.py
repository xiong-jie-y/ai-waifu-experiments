import datetime
import difflib
from typing import Text
from git import Repo
import os
import subprocess
from collections import defaultdict
import yaml
import json
import copy
import random
import click
from dataclasses import dataclass

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtGui import QImage, QPalette, QPixmap
from PyQt5.QtGui import QPainter, QFont, QColor
from PyQt5.QtCore import QThread, Qt

@dataclass
class Praise:
    text: str
    score: float = 0.0


MONITOR_CONFIG = "monitor_config.yaml"

config = yaml.load(open(MONITOR_CONFIG), Loader=yaml.FullLoader)


LAST_STATUS_FILE = "last_status.yaml"
GIT_LOCAL_ROOT = os.path.expanduser("~/data")
ACHIVEMENT_FILE = "achivements.json"
PRAISED_TIMESTAMP_FILE = 'praised_timestamp.json'


class TextMetric:
    def add_diff(self, diff_item):
        if diff_item.change_type == 'M':
            diff = difflib.ndiff(
                diff_item.a_blob.data_stream.read().decode('utf-8').split("\n"),
                diff_item.b_blob.data_stream.read().decode('utf-8').split("\n"))
            self.process_diff(diff)
        elif diff_item.change_type == 'A':
            self.process_addition(
                diff_item.b_blob.data_stream.read().decode('utf-8').split("\n"))
        else:
            print(diff_item.change_type)


class LineCountMetric(TextMetric):
    def __init__(self):
        self.value = 0

    def process_diff(self, diff):
        for l in diff:
            if l.startswith("+ "):
                self.value += 1

    def process_addition(self, data):
        self.value += len(data)


class CharacterCountMetric(TextMetric):
    def __init__(self):
        self.value = 0

    def process_diff(self, diff):
        for l in diff:
            if l.startswith("+ "):
                self.value += len(l) - 2

    def process_addition(self, data):
        for line in data:
            self.value += len(line)


METRIC_MAP = {
    ".md": CharacterCountMetric(),
    ".py": LineCountMetric(),
}


class TextPraisingAgent():
    def __init__(self):
        self.last_status = yaml.load(
            open(LAST_STATUS_FILE), Loader=yaml.FullLoader)
        # print(self.last_status)
        self.achievements = json.load(open(ACHIVEMENT_FILE))
        self.praised_timestamp = json.load(open(PRAISED_TIMESTAMP_FILE))

    def check_repositories_update(self):
        addition_lines = {}
        for git_repository in config['monitor_targets']['git_repositories']:
            git_local_path = os.path.join(
                GIT_LOCAL_ROOT, git_repository['name'])
            if not os.path.exists(git_local_path):
                clone_command = f"git clone {git_repository['url']} {git_local_path}"
                subprocess.call(clone_command, shell=True)

            repo = Repo(git_local_path)
            o = repo.remotes.origin
            o.pull()

            latest_commit = repo.head.commit
            last_commits = self.last_status['last_commits'] if 'last_commits' in self.last_status else {
            }
            if git_repository['name'] not in last_commits:
                last_commits[git_repository['name']] = latest_commit.hexsha

            diff_index = repo.commit(
                last_commits[git_repository['name']]).diff(latest_commit)
            # import IPython; IPython.embed()
            # diff_index = repo.commit('HEAD~2').diff(t)
            for diff_item in diff_index:
                try:
                    ext_name = os.path.splitext(diff_item.b_path)[1]
                    if ext_name not in METRIC_MAP:
                        continue

                    if ext_name not in addition_lines:
                        addition_lines[ext_name] = METRIC_MAP[ext_name]

                    addition_lines[ext_name].add_diff(diff_item)

                except:
                    import IPython
                    IPython.embed()

            last_commits[git_repository['name']] = latest_commit.hexsha

            # print(addition_lines)

        return dict(addition_lines)

    def _try_to_praise(self, changes, serifu_signal):
        num_doc_lines = changes['.md'].value if '.md' in changes else 0
        num_code_lines = changes['.py'].value if '.py' in changes else 0

        now_str = datetime.datetime.now().isoformat()

        last_achievement = self.achievements[-1]
        praises = []
        # Praise with absolute values.
        if num_doc_lines > 1000 and num_code_lines > 100:
            praises.append(Praise("すっごーい！", score=1.0))
        if num_code_lines > 100:
            praises.append(
                Praise(f"たくさんコードを書いたね！えらい！{num_code_lines}行も書いてる！", score=0.5))
        if num_doc_lines > 500:
            praises.append(Praise(f"たくさん文章を書いたね！{num_doc_lines}も書いてる！", score=0.5))

        # Praises with diffs.
        diff_code_lines = num_code_lines - last_achievement['num_code_lines']
        if diff_code_lines > 0:
            praises.append(Praise(f"1時間前より、{diff_code_lines}文字もたくさん書いてるよ！", score=0.5))
        diff_doc_lines = num_doc_lines - last_achievement['num_doc_lines']
        if diff_doc_lines > 0:
            praises.append(Praise(f"1時間前より、{diff_doc_lines}文字もたくさん書いてるよ！", score=0.5))
        
        # Unconditional praises.
        praises.append(Praise("頑張ってるね！！"))

        praises.sort(key=lambda p: p.score)
        # random.shuffle(praises)
        praise = None
        for p in praises:
            if p.text not in self.praised_timestamp or \
                (datetime.datetime.now() - datetime.datetime.fromisoformat(self.praised_timestamp[p.text])) \
                    > datetime.timedelta(days=30):
                praise = p
                self.praised_timestamp[p.text] = datetime.datetime.now().isoformat()

        if praise is None:
            praise_txt = "今日はいい天気だね〜〜"
        else:
            praise_txt = praise.text

        print(f"{now_str}:{praise_txt}")
        serifu_signal.emit(praise_txt)

        achievement_record = {}
        achievement_record['num_doc_lines'] = num_doc_lines
        achievement_record['num_code_lines'] = num_code_lines
        achievement_record['date'] = datetime.datetime.now().isoformat()
        self.achievements.append(achievement_record)

    def try_to_praise(self, serifu_signal):
        changes = self.check_repositories_update()
        # print(changes)
        self._try_to_praise(changes, serifu_signal)

    def save_state(self):
        yaml.dump(self.last_status, open(LAST_STATUS_FILE, 'w'))
        json.dump(self.achievements, open(ACHIVEMENT_FILE, 'w'))
        json.dump(self.praised_timestamp, open(PRAISED_TIMESTAMP_FILE, 'w', encoding='utf-8'))


def try_to_praise(serifu_signal):
    agent = TextPraisingAgent()
    agent.try_to_praise(serifu_signal)
    agent.save_state()

class ConcurrentlyWorker(QThread):
    def __init__(self, with_scheduler, serifu_signal):
        super().__init__()
        self.with_scheduler = with_scheduler
        self.serifu_signal = serifu_signal

    def run(self):
        if self.with_scheduler:
            from apscheduler.schedulers.blocking import BlockingScheduler

            scheduler = BlockingScheduler()
            scheduler.add_job(try_to_praise, 'interval', hours=1, args=(self.serifu_signal,))
            scheduler.start()
        else:
            try_to_praise(self.serifu_signal)


import multiprocessing

from PyQt5.QtCore import QObject, pyqtSignal

class SerifuWindow(QWidget):
    serifu_signal = pyqtSignal(str)

    def __init__(self, with_scheduler):
        super().__init__()
        self.setWindowTitle('Image View')

        image = QImage('data/fukidashi.jpg')
        painter = QPainter()
        painter = painter
        painter.begin(image)
        painter.setPen(Qt.black)
        painter.setFont(QFont('Times', 50))
        painter.drawText(image.rect(), Qt.AlignCenter, 'わーい！')
        painter.end()


        self.imageLabel = QLabel()
        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        self.imageLabel.scaleFactor = 1.0
        layout = QVBoxLayout()
        layout.addWidget(self.imageLabel)
        self.setLayout(layout)
        self.resize(400, 300)

            # self.painter.drawText(self.image.rect(), Qt.AlignCenter, text)

        self.serifu_signal.connect(self.change_serifu)

        # p = multiprocessing.Process(target=run_checker, args=(with_scheduler,self.serifu_signal))
        thread = ConcurrentlyWorker(with_scheduler, self.serifu_signal)
        thread.start()

        self.show()

    def change_serifu(self, text):
        print("aho")
        print(text)

        image = QImage('data/fukidashi.jpg')
        painter = QPainter()
        painter = painter
        painter.begin(image)
        painter.setPen(Qt.black)
        painter.setFont(QFont('Times', 50))
        painter.drawText(image.rect(), Qt.AlignCenter, text)
        painter.end()

        self.imageLabel.setPixmap(QPixmap.fromImage(image))

@click.command()
@click.option("--with-scheduler", is_flag=True)
def main(with_scheduler):
    app = QApplication([])
    win = SerifuWindow(with_scheduler)
    app.exec_()

if __name__ == "__main__":
    main()
