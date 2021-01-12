import difflib
from typing import Text
from git import Repo
import os
import subprocess
from collections import defaultdict
import yaml
import json
import copy
import datetime

MONITOR_CONFIG = "monitor_config.yaml"

config = yaml.load(open(MONITOR_CONFIG), Loader=yaml.FullLoader)


LAST_STATUS_FILE = "last_status.yaml"
GIT_LOCAL_ROOT = os.path.expanduser("~/data")
ACHIVEMENT_FILE = "achivements.json"

class TextPraisingAgent():
    def __init__(self):
        self.last_status = yaml.load(open(LAST_STATUS_FILE), Loader=yaml.FullLoader)
        # print(self.last_status)
        self.achievements = json.load(open(ACHIVEMENT_FILE))

    def check_repositories_update(self):
        addition_lines = defaultdict(lambda: 0)
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
            last_commits = self.last_status['last_commits'] if 'last_commits' in self.last_status else {}
            if git_repository['name'] not in last_commits:
                last_commits[git_repository['name']] = latest_commit.hexsha

            diff_index = repo.commit(
                last_commits[git_repository['name']]).diff(latest_commit)
            # import IPython; IPython.embed()
            # diff_index = repo.commit('HEAD~2').diff(t)
            for diff_item in diff_index:
                ext_name = os.path.splitext(diff_item.b_path)[1]
                if diff_item.change_type == 'M':
                    diff = difflib.ndiff(
                        diff_item.a_blob.data_stream.read().decode('utf-8').split("\n"),
                        diff_item.b_blob.data_stream.read().decode('utf-8').split("\n"))
                    for l in diff:
                        if l.startswith("+ "):
                            addition_lines[ext_name] += 1
                elif diff_item.change_type == 'A':
                    addition_lines[ext_name] += len(
                        diff_item.b_blob.data_stream.read().decode('utf-8').split("\n"))
                else:
                    print(diff_item.change_type)
            
            last_commits[git_repository['name']] = latest_commit.hexsha

            # print(addition_lines)

        return dict(addition_lines)

    def _try_to_praise(self, changes):
        num_doc_lines = changes['.md'] if '.md' in changes else 0
        num_code_lines = changes['.py'] if '.py' in changes else 0

        now_str = datetime.datetime.now().isoformat()

        praise = ""
        if num_doc_lines > 500 and num_code_lines > 100:
            praise = "すっごーい！"
        elif num_code_lines > 100:
            praise = f"たくさんコードを書いたね！えらい！{num_code_lines}行も書いたんだよ！"
        elif num_doc_lines > 500:
            praise = "たくさん文章を書いたね！"
        else:
            praise = "頑張ってるね！！"

        print(f"{now_str}:{praise}")

        last_achievement = self.last_status['last_achievement'] if 'last_achievement' in self.last_status else {}
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

try_to_praise()

# from apscheduler.schedulers.blocking import BlockingScheduler

# scheduler = BlockingScheduler()
# scheduler.add_job(try_to_praise, 'interval', hours=1)
# scheduler.start()
