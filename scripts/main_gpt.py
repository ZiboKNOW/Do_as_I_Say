import os

import numpy as np
import threading
import copy
import openai
import cv2
import sys
# from google.colab.patches import cv2_imshow
# from moviepy.editor import ImageSequenceClip
script_path = os.path.dirname(os.path.realpath(__file__))

# 获取上级目录的路径
parent_path = os.path.abspath(os.path.join(script_path, os.pardir))

# 添加上级目录到sys.path
sys.path.append(parent_path)
import astunparse
import shapely
import ast
from time import sleep
from shapely.geometry import *
from shapely.affinity import *
from openai.error import RateLimitError, APIConnectionError
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from Units.gpt import ENV_API
openai.api_key = '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'
os.environ['HTTP_PROXY'] = os.environ['http_proxy'] = "http://127.0.0.1:7890"  
os.environ['HTTPS_PROXY'] = os.environ['https_proxy'] = 'http://127.0.0.1:7890'  
model_name = 'gpt-3.5-turbo' # 'text-davinci-002'
class LMP:

    def __init__(self, name, cfg, lmp_fgen, fixed_vars, variable_vars):
        self._name = name
        self._cfg = cfg

        self._base_prompt = self._cfg['prompt_text']
        self._base_prompt_title = self._cfg['prompt_text_title']
        self._stop_tokens = list(self._cfg['stop'])

        self._lmp_fgen = lmp_fgen

        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_prompt(self, query, context=''):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self._variable_vars.keys())}"
        else:
            variable_vars_imports_str = ''
        prompt_title = self._base_prompt_title.replace('{variable_vars_imports}', variable_vars_imports_str)
        if self._cfg['maintain_session']:
            prompt_title += f'\n{self.exec_hist}'
        messages.append({"role": "user", "content": prompt_title})
        
        list_text =  self._base_prompt.splitlines()
        for key, n in enumerate(list_text):
            if n.startswith('objects') and key == 0:
                messages.append({"role": "user", "content": n +'\n'+list_text[key+1]})
                last_idx = key
            elif n.startswith('objects') and key != 0:
                sub_list = list_text[last_idx+2:key]
                result = '\n'.join(sub_list)
                messages.append({"role": "assistant", "content":result})
                messages.append({"role": "user", "content": n +'\n'+list_text[key+1]})
                last_idx = key
        sub_list = list_text[last_idx+2:]
        result = '\n'.join(sub_list)
        messages.append({"role": "assistant", "content":result})
        
        user_prompt = ''
        if context != '':
            user_prompt += f'\n{context}'
        use_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        user_prompt += f'\n{use_query}'
        messages.append({"role": "user", "content": user_prompt})

        return messages, use_query

    def __call__(self, query, context='', **kwargs):
        prompt, use_query = self.build_prompt(query, context=context)

        while True:
            try:
                code_str = openai.ChatCompletion.create(
                    model=self._cfg['engine'],
                    messages=prompt,
                    stop=self._stop_tokens,
                    temperature=self._cfg['temperature'],
                    max_tokens=self._cfg['max_tokens']
                )['choices'][0]['message']['content'].strip()
                break
            except (RateLimitError, APIConnectionError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 10s.')
                sleep(10)

        if self._cfg['include_context'] and context != '':
            to_exec = f'{context}\n{code_str}'
            to_log = f'{context}\n{use_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{use_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())
        print(f'LMP {self._name} exec:\n\n{to_log_pretty}\n')

        new_fs = self._lmp_fgen.create_new_fs_from_code(code_str)
        self._variable_vars.update(new_fs)

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        if not self._cfg['debug_mode']:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_exec}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        if self._cfg['has_return']:
            return lvars[self._cfg['return_val_name']]


class LMPFGen:

    def __init__(self, cfg, fixed_vars, variable_vars):
        self._cfg = cfg

        self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars

        self._base_prompt = self._cfg['prompt_text']
        self._base_prompt_title = self._cfg['prompt_text_title']
    def create_f_from_sig(self, f_name, f_sig, other_vars=None, fix_bugs=False, return_src=False):
        print(f'Creating function: {f_sig}')
        
        use_query = f'{self._cfg["query_prefix"]}{f_sig}{self._cfg["query_suffix"]}'
        prompt = self.build_gen_prompt(use_query)
        print("prompt: ",prompt)
        while True:
            try:
                f_src = openai.ChatCompletion.create(
                    model=self._cfg['engine'],
                    messages=prompt,
                    stop=self._stop_tokens,
                    temperature=self._cfg['temperature'],
                    max_tokens=self._cfg['max_tokens']
                )['choices'][0]['message']['content'].strip()
                break
            except (RateLimitError, APIConnectionError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 10s.')
                sleep(10)

        if fix_bugs:
            f_src = openai.Edit.create(
                model='code-davinci-edit-001',
                input='# ' + f_src,
                temperature=0,
                instruction='Fix the bug if there is one. Improve readability. Keep same inputs and outputs. Only small changes. No comments.',
            )['choices'][0]['text'].strip()

        if other_vars is None:
            other_vars = {}
        gvars = merge_dicts([self._fixed_vars, self._variable_vars, other_vars])
        lvars = {}
        if not self._cfg['debug_mode']:
            exec_safe(f_src, gvars, lvars)

        f = lvars[f_name]

        to_print = highlight(f'{use_query}\n{f_src}', PythonLexer(), TerminalFormatter())
        print(f'LMP FGEN created:\n\n{to_print}\n')

        if return_src:
            return f, f_src
        return f
    
    def build_gen_prompt(self,use_query):
        messages = [{"role": "system", "content": "You are a helpful assistant to write code to plan the trajectory of a drone to finish a task"}]
        messages.append({"role": "user", "content": self._base_prompt_title})
        list_text = self._base_prompt.splitlines()
        for key, n in enumerate(list_text):
            if n.startswith('#') and key == 0:
                messages.append({"role": "user", "content": n})
                last_idx = key
            elif n.startswith('#') and key != 0:
                sub_list = list_text[last_idx+1:key]
                result = '\n'.join(sub_list)
                messages.append({"role": "assistant", "content":result})
                messages.append({"role": "user", "content": n})
                last_idx = key
        sub_list = list_text[last_idx+1:]
        result = '\n'.join(sub_list)
        messages.append({"role": "assistant", "content":result})
        messages.append({"role": "user", "content":use_query})
        return messages
        
    def create_new_fs_from_code(self, code_str, other_vars=None, fix_bugs=False, return_src=False):
        fs, f_assigns = {}, {}
        f_parser = FunctionParser(fs, f_assigns)
        f_parser.visit(ast.parse(code_str))
        for f_name, f_assign in f_assigns.items():
            if f_name in fs:
                fs[f_name] = f_assign

        if other_vars is None:
            other_vars = {}

        new_fs = {}
        srcs = {}
        for f_name, f_sig in fs.items():
            all_vars = merge_dicts([self._fixed_vars, self._variable_vars, new_fs, other_vars])
            # print(all_vars.keys())
            if not var_exists(f_name, all_vars):
                f, f_src = self.create_f_from_sig(f_name, f_sig, new_fs, fix_bugs=fix_bugs, return_src=True)

                # recursively define child_fs in the function body if needed
                f_def_body = astunparse.unparse(ast.parse(f_src).body[0].body)
                child_fs, child_f_srcs = self.create_new_fs_from_code(
                    f_def_body, other_vars=all_vars, fix_bugs=fix_bugs, return_src=True
                )

                if len(child_fs) > 0:
                    new_fs.update(child_fs)
                    srcs.update(child_f_srcs)

                    # redefine parent f so newly created child_fs are in scope
                    gvars = merge_dicts([self._fixed_vars, self._variable_vars, new_fs, other_vars])
                    lvars = {}
                    
                    exec_safe(f_src, gvars, lvars)
                    
                    f = lvars[f_name]

                new_fs[f_name], srcs[f_name] = f, f_src

        if return_src:
            return new_fs, srcs
        return new_fs


class FunctionParser(ast.NodeTransformer):

    def __init__(self, fs, f_assigns):
      super().__init__()
      self._fs = fs
      self._f_assigns = f_assigns

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            f_sig = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.func).strip()
            self._fs[f_name] = f_sig
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            assign_str = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.value.func).strip()
            self._f_assigns[f_name] = assign_str
        return node


def var_exists(name, all_vars):
    try:
        eval(name, all_vars)
    except:
        exists = False
    else:
        exists = True
    return exists


def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }
    

def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str
  
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    exec(code_str, custom_gvars, lvars)
    
    
    
    
    
# class LMP_wrapper():

#   def __init__(self, cfg, render=False):
#     # self.env = env
#     self._cfg = cfg
#     # self.object_names = list(self._cfg['env']['init_objs'])
    
#     # self._min_xy = np.array(self._cfg['env']['coords']['bottom_left'])
#     # self._max_xy = np.array(self._cfg['env']['coords']['top_right'])
#     # self._range_xy = self._max_xy - self._min_xy

#     # self._table_z = self._cfg['env']['coords']['table_z']
#     # self.render = render

#   def is_obj_visible(self, obj_name):
#     return obj_name in self.object_names

#   def get_obj_names(self):
#     return ['person in white', 'person in blue', 'person in pink', 'green SUV', 'person with umbrella', 'white car',  'person in black', 'black MPV']

#   def get_obj_pos(self, obj_name):
#     # return the xy position of the object in robot base frame
#     # return self.env.get_obj_pos(obj_name)[:2]
#     return

#   def get_obj_position_np(self, obj_name):
#     return self.get_pos(obj_name)

#   def get_bbox(self, obj_name):
#     # return the axis-aligned object bounding box in robot base frame (not in pixels)
#     # the format is (min_x, min_y, max_x, max_y)
#     # bbox = self.env.get_bounding_box(obj_name)
#     # return bbox
#     return

#   def follow_traj(self, traj):
#     for pos in traj:
#       self.goto_pos(pos)

#   def move_to_pos(self, position):
#     print(f'move to {position}')

#   def get_ego_pos(self):
#     print('get ego pose')
#     return

prompt_tabletop_ui_title = '''
# Python 3D drone flying ctonrol script
import numpy as np
from env_utils import get_obj_pos, get_obj_names, say, is_obj_visible, move_to_pos, get_ego_pos
from plan_utils import parse_obj_name, parse_position
'''.strip()

prompt_tabletop_ui = '''
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van']
# fly to the person in red.
say('Ok - flying to the person in red')
target_pos = get_obj_pos('person in red')
move_to_pos(target_pos)
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van']
# where did you go.
say('I moved to person in red')
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van']
# move forward.
say('Got it - moving forward')
target_pos = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
move_to_pos(target_pos)
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van']
# move forward 1m.
say('Got it - moving forward')
target_pos = [1, 0.0, 0.0, 0.0, 0.0, 0.0]
move_to_pos(target_pos)
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van']
# move higher.
say('Got it - moving higher')
target_pos = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
move_to_pos(target_pos)
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van']
#fly to the left.
say('Got it - moving to the left')
target_pos = [0, 0.1, 0.0, 0.0, 0.0, 0.0]
move_to_pos(target_pos)
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van']
#fly to the left 1m.
say('Got it - moving to the left')
target_pos = [0, 1, 0.0, 0.0, 0.0, 0.0]
move_to_pos(target_pos)
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van']
# fly to the person in purple clothes
person_name = parse_obj_name('person in purple clothes', f'objects = {get_obj_names()}')
say(f'Sorry - there is no person in purple, but there is {"".join(person_name)}')
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van']
# Fly to the person closest to you
person_name = parse_obj_name('the person closest to you', f'objects = {get_obj_names()}')
say(f'No problem! flying to {"".join(person_name)}')
target_pos = get_obj_pos(person_name)
move_to_pos(target_pos)
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van',  'person in yellow' ]
# fly to the person with small banana colored clothes.
person_name = parse_obj_name('the person with small banana colored clothes', f'objects = {get_obj_names()}')
say(f'No problem! flying to {", ".join(person_name)}')
target_pos = get_obj_pos(person_name)
move_to_pos(target_pos)
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van',  'person in yellow' ]
# turn left.
say(f'No problem!')
target_pos = [0.0, 0.0, 0.0, 10.0, 0.0, 0.0]
move_to_pos(target_pos)
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van',  'person in yellow' ]
# turn 30 degrees to the left
say(f'No problem!')
target_pos = [0.0, 0.0, 0.0, 30.0, 0.0, 0.0]
move_to_pos(target_pos)
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van',  'person in yellow' ]
# turn 30 degrees to the right
say(f'No problem!')
target_pos = [0.0, 0.0, 0.0, - 30.0, 0.0, 0.0]
move_to_pos(target_pos)
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van',  'person in yellow' ]
# if you see a person with yellow clothes fly to it
if is_obj_visible('person in yellow'):
  say('flying to the person with yellow clothes')
  target_pos = get_obj_pos('person in yellow')
  move_to_pos(target_pos)
else:
  say('I don\'t see a person in yellow')
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van',  'person in yellow' ]
# fly to the person closest to the green car
say(f'No problem!')
car_name = parse_obj_name('green car', f'objects = {get_obj_names()}')
person_name = parse_obj_name(f'the person closest to the {car_name}',f'objects = {get_obj_names()}')
target_pos =  get_obj_pos(person_name)
move_to_pos(target_pos)
'''.strip()



prompt_parse_obj_name_title ='''import numpy as np
from env_utils import get_obj_pos, parse_position
from utils import get_obj_positions_np'''.strip()

prompt_parse_obj_name = '''
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van',  'person in yellow']
# the person closest to the green car.
person_names = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'person in yellow']
person_positions = get_obj_positions_np(person_names)
closest_person_idx = get_closest_idx(points=person_positions, point=get_obj_pos('green van'))
closest_person_name = person_names[closest_person_idx]
ret_val = closest_person_name
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van',  'person in yellow']
# the persons.
ret_val = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'person in yellow']
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van',  'person in yellow']
# the green car.
ret_val = 'green van'
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van',  'person in yellow']
# a person that's not in blue
person_names = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'person in yellow']
for person_name in person_names:
    if person_name != 'person in blue':
        ret_val = person_name
objects = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'white car', 'green van',  'person in yellow']
# persons in the front of the white car.
person_names = ['person in red', 'person in blue', 'person in pink', 'person with umbrella', 'person in yellow']
white_car_pos = get_obj_pos('white car')
use_person_names = []
for person_name in person_names:
    if get_obj_pos(person_name)[0] > white_car_pos[0]:
        use_person_names.append(person_name)
ret_val = use_person_names
objects = ['person in white', 'person in blue', 'person in pink', 'green SUV', 'person with umbrella', 'white car',  'person in yellow']
# the person with umbrella.
ret_val = 'person with umbrella'
objects = ['person in white', 'person in blue', 'person in pink', 'green SUV', 'person with umbrella', 'white car',  'person in black', 'black MPV']
# the left most car.
car_names = ['white car',  'green SUV', 'black MPV']
car_positions = get_obj_positions_np(car_names)
left_car_idx = np.argsort(block_positions[:, 0])[-1]
left_car_name = car_names[left_car_idx]
ret_val = left_car_name
objects = ['person in white', 'person in blue', 'person in pink', 'green SUV', 'person with umbrella', 'white car',  'person in black', 'black MPV']
# the person closest to you.
person_names = ['person in white', 'person in blue', 'person in pink','person with umbrella', 'person in black']
person_positions = get_obj_positions_np(person_names)
ego_position = get_ego_pos()
closest_person_idx = get_closest_idx(points=person_positions, point=ego_position)
closest_person_name = person_names[closest_person_idx]
ret_val = closest_person_name
objects = ['person in white', 'person in blue', 'person in pink', 'green SUV', 'person with umbrella', 'white car',  'person in black', 'black MPV']
# the third person from the right.
person_names = ['person in white', 'person in blue', 'person in pink','person with umbrella', 'person in black']
person_positions = get_obj_positions_np(person_names)
person_idx = np.argsort(block_positions[:, 0])[2]
person_name = person_names[person_idx]
ret_val = person_name
'''.strip()

prompt_parse_position = '''
import numpy as np
from shapely.geometry import *
from shapely.affinity import *
from env_utils import denormalize_xy, parse_obj_name, get_obj_names, get_obj_pos

# a 30cm horizontal line in the middle with 3 points.
middle_pos = denormalize_xy([0.5, 0.5]) 
start_pos = middle_pos + [-0.3/2, 0]
end_pos = middle_pos + [0.3/2, 0]
line = make_line(start=start_pos, end=end_pos)
points = interpolate_pts_on_line(line=line, n=3)
ret_val = points
# a 20cm vertical line near the right with 4 points.
middle_pos = denormalize_xy([1, 0.5]) 
start_pos = middle_pos + [0, -0.2/2]
end_pos = middle_pos + [0, 0.2/2]
line = make_line(start=start_pos, end=end_pos)
points = interpolate_pts_on_line(line=line, n=4)
ret_val = points
# a diagonal line from the top left to the bottom right corner with 5 points.
top_left_corner = denormalize_xy([0, 1])
bottom_right_corner = denormalize_xy([1, 0])
line = make_line(start=top_left_corner, end=bottom_right_corner)
points = interpolate_pts_on_line(line=line, n=5)
ret_val = points
# a triangle with size 10cm with 3 points.
polygon = make_triangle(size=0.1, center=denormalize_xy([0.5, 0.5]))
points = get_points_from_polygon(polygon)
ret_val = points
# the corner closest to the sun colored block.
block_name = parse_obj_name('the sun colored block', f'objects = {get_obj_names()}')
corner_positions = np.array([denormalize_xy(pos) for pos in [[0, 0], [0, 1], [1, 1], [1, 0]]])
closest_corner_pos = get_closest_point(points=corner_positions, point=get_obj_pos(block_name))
ret_val = closest_corner_pos
# the side farthest from the right most bowl.
bowl_name = parse_obj_name('the right most bowl', f'objects = {get_obj_names()}')
side_positions = np.array([denormalize_xy(pos) for pos in [[0.5, 0], [0.5, 1], [1, 0.5], [0, 0.5]]])
farthest_side_pos = get_farthest_point(points=side_positions, point=get_obj_pos(bowl_name))
ret_val = farthest_side_pos
# a point above the third block from the bottom.
block_name = parse_obj_name('the third block from the bottom', f'objects = {get_obj_names()}')
ret_val = get_obj_pos(block_name) + [0.1, 0]
# a point 10cm left of the bowls.
bowl_names = parse_obj_name('the bowls', f'objects = {get_obj_names()}')
bowl_positions = get_all_object_positions_np(obj_names=bowl_names)
left_obj_pos = bowl_positions[np.argmin(bowl_positions[:, 0])] + [-0.1, 0]
ret_val = left_obj_pos
# the bottom side.
bottom_pos = denormalize_xy([0.5, 0])
ret_val = bottom_pos
# the top corners.
top_left_pos = denormalize_xy([0, 1])
top_right_pos = denormalize_xy([1, 1])
ret_val = [top_left_pos, top_right_pos]
'''.strip()

prompt_fgen_title = '''
import numpy as np
from shapely.geometry import *
from shapely.affinity import *

from env_utils import get_obj_pos, get_obj_names
from ctrl_utils import put_first_on_second
'''.strip()

prompt_fgen = '''
# define function: total = get_total(xs=numbers).
def get_total(xs):
    return np.sum(xs)

# define function: y = eval_line(x, slope, y_intercept=0).
def eval_line(x, slope, y_intercept):
    return x * slope + y_intercept

# define function: pt = get_pt_to_the_left(pt, dist).
def get_pt_to_the_left(pt, dist):
    return pt + [-dist, 0]

# define function: pt = get_pt_to_the_top(pt, dist).
def get_pt_to_the_top(pt, dist):
    return pt + [0, dist]

# define function line = make_line_by_length(length=x).
def make_line_by_length(length):
  line = LineString([[0, 0], [length, 0]])
  return line

# define function: line = make_vertical_line_by_length(length=x).
def make_vertical_line_by_length(length):
  line = make_line_by_length(length)
  vertical_line = rotate(line, 90)
  return vertical_line

# define function: pt = interpolate_line(line, t=0.5).
def interpolate_line(line, t):
  pt = line.interpolate(t, normalized=True)
  return np.array(pt.coords[0])

# example: scale a line by 2.
line = make_line_by_length(1)
new_shape = scale(line, xfact=2, yfact=2)

# example: put object1 on top of object0.
put_first_on_second('object1', 'object0')

# example: get the position of the first object.
obj_names = get_obj_names()
pos_2d = get_obj_pos(obj_names[0])
'''.strip()

cfg_tabletop = {
  'lmps': {
    'tabletop_ui': {
      'prompt_text': prompt_tabletop_ui,
      'prompt_text_title': prompt_tabletop_ui_title,
      'engine': model_name,
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#', 'objects = ['],
      'maintain_session': True,
      'debug_mode': False,
      'include_context': True,
      'has_return': False,
      'return_val_name': 'ret_val',
    },
    'parse_obj_name': {
      'prompt_text': prompt_parse_obj_name,
      'prompt_text_title': prompt_parse_obj_name_title,
      'engine': model_name,
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#', 'objects = ['],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
      'has_return': True,
      'return_val_name': 'ret_val',
    },
    'parse_position': {
      'prompt_text': prompt_parse_position,
      'engine': model_name,
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#'],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
      'has_return': True,
      'return_val_name': 'ret_val',
    },
    'fgen': {
      'prompt_text': prompt_fgen,
      'prompt_text_title': prompt_fgen_title,
      'engine': model_name,
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# define function: ',
      'query_suffix': '.',
      'stop': ['# define', '# example'],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
    }
  }
}

def setup_LMP(cfg_tabletop):
  # LMP env wrapper
  cfg_tabletop = copy.deepcopy(cfg_tabletop)
#   cfg_tabletop['env'] = dict()
#   cfg_tabletop['env']['init_objs'] = list(env.obj_name_to_id.keys())
  # cfg_tabletop['env']['coords'] = lmp_tabletop_coords
  LMP_env = ENV_API()
  thread = threading.Thread(target=LMP_env.run)
  thread.start()
  # creating APIs that the LMPs can interact with
  fixed_vars = {
      'np': np
  }
  fixed_vars.update({
      name: eval(name)
      for name in shapely.geometry.__all__ + shapely.affinity.__all__
  })
#   variable_vars = {
#       k: getattr(LMP_env, k)
#       for k in [
#           'get_bbox', 'get_obj_pos', 'is_obj_visible', 'get_obj_names',
#           'move_to_pos','get_ego_pos','get_obj_position_np'
#       ]
#   }
  variable_vars = {
        k: getattr(LMP_env, k)
        for k in [
            'move_to_pos','get_ego_pos','get_obj_pos','get_obj_names'
        ]
  }
  variable_vars['say'] = lambda msg: print(f'robot says: {msg}')

  # creating the function-generating LMP
  lmp_fgen = LMPFGen(cfg_tabletop['lmps']['fgen'], fixed_vars, variable_vars)

  # creating other low-level LMPs
  variable_vars.update({
      k: LMP(k, cfg_tabletop['lmps'][k], lmp_fgen, fixed_vars, variable_vars)
      for k in ['parse_obj_name']
  })

  # creating the LMP that deals w/ high-level language commands
  lmp_tabletop_ui = LMP(
      'tabletop_ui', cfg_tabletop['lmps']['tabletop_ui'], lmp_fgen, fixed_vars, variable_vars
  )
  
  return lmp_tabletop_ui
if __name__ == '__main__':
    lmp_tabletop_ui = setup_LMP(cfg_tabletop)

    # while True:
    #     if env_api.got_ego_pose:
    #         sting_list = input()
    #         list_inpot = sting_list.split()
    #         list_inpot = [float(x) for x in list_inpot]
    #         env_api.move_to_pos(list_inpot)
    #     else:
    #         print('without ego position')
    while True:
        user_input = input() #@param {allow-input: true, type:"string"}
        object_list = ['law office', 'green tree', 'salon', 'black suv', 'standing person in white', 'walking person in white']

        lmp_tabletop_ui(user_input, f'objects = {object_list}')
