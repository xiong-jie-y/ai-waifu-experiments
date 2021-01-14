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

MONITOR_CONFIG = "monitor_config.yaml"

config = yaml.load(open(MONITOR_CONFIG), Loader=yaml.FullLoader)


LAST_STATUS_FILE = "last_status.yaml"
GIT_LOCAL_ROOT = os.path.expanduser("~/data")
ACHIVEMENT_FILE = "achivements.json"


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

    def _try_to_praise(self, changes):
        num_doc_lines = changes['.md'].value if '.md' in changes else 0
        num_code_lines = changes['.py'].value if '.py' in changes else 0

        now_str = datetime.datetime.now().isoformat()

        praise = ""
        if num_doc_lines > 1000 and num_code_lines > 100:
            praise = "すっごーい！"
        elif num_code_lines > 100:
            praise = f"たくさんコードを書いたね！えらい！{num_code_lines}行も書いてる！"
        elif num_doc_lines > 500:
            praise = "たくさん文章を書いたね！"
        else:
            praise = "頑張ってるね！！"

        print(f"{now_str}:{praise}")

        last_achievement = self.last_status['last_achievement'] if 'last_achievement' in self.last_status else {
        }
        last_achievement['num_doc_lines'] = num_doc_lines
        last_achievement['num_code_lines'] = num_code_lines
        achievement_record = copy.deepcopy(last_achievement)
        achievement_record['date'] = datetime.datetime.now().isoformat()
        self.achievements.append(achievement_record)

    def try_to_praise(self):
        changes = self.check_repositories_update()
        # print(changes)
        self._try_to_praise(changes)

    def save_state(self):
        yaml.dump(self.last_status, open(LAST_STATUS_FILE, 'w'))
        json.dump(self.achievements, open(ACHIVEMENT_FILE, 'w'))


def try_to_praise():
    agent = TextPraisingAgent()
    agent.try_to_praise()
    agent.save_state()


def main():
    # try_to_praise()
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler()
    scheduler.add_job(try_to_praise, 'interval', hours=1)
    scheduler.start()


if __name__ == "__main__":
    main()
