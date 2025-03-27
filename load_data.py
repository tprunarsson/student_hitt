from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Literal
import json
import pandas as pd

def load_enriched_schedule(json_path="data/schedule_enriched.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []

    for lcid, course in raw.items():
        course_id = course.get("course_id")
        full_name = course.get("full_name")
        short_name = course.get("short_name")
        teachers = ", ".join(course.get("teachers", []))
        schedule = course.get("schedule", {})

        for room_label, room in schedule.items():
            room_info = room.get("room_info", {})
            location = room_info.get("name") or room_label
            building = room_info.get("homeprogram", "Unknown")

            for weekday, slots in room.get("slots", {}).items():
                weekday_int = {
                    "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5
                }.get(weekday.lower(), None)
                if weekday_int is None:
                    continue

                for slot in slots:
                    rows.append({
                        "lcid": lcid,
                        "cid": short_name,
                        "name": full_name,
                        "teacher": teachers,
                        "weekday": weekday_int,
                        "start": slot["start_time"] + ":00",
                        "end": slot["end_time"] + ":00",
                        "location": location,
                        "building": building,
                        "start_week": slot["start_week"],
                        "end_week": slot["end_week"],
                        "lecture": slot.get("lecture", True),  # ← Add this line
                        # Add these 👇
                        "sidcount": course.get("sidcount"),
                        "level": course.get("level"),
                        "season": course.get("season"),
                        "nameeng": course.get("nameeng")
                    })

    df = pd.DataFrame(rows)
    return df


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    room_df = pd.read_csv('student_hitt/data/rooms.csv')
    fid_df = pd.read_csv('student_hitt/data/field_courses.csv')
    clash_df = pd.read_csv('student_hitt/data/clashes.csv')
    name_df = pd.read_csv('student_hitt/data/course_names.csv')
    schedule_df = load_enriched_schedule('student_hitt/data/schedule_enriched.json')
    return room_df, fid_df, clash_df, name_df, schedule_df


@dataclass
class Compatibility:
    can_share_slot: Literal['yes', 'no', 'avoid']
    reason: str
    data: dict[str, list] 
    pairs: set[tuple[str, str]] = field(default_factory=set, init=False)

    def __post__init(self):
        for k, v in self.data.items():
            self.add(k, v)

    def add(self, fid_id: str, courses: list[str]):
        self.data[fid_id] = courses
        unique_courses = set(courses)
        course_pairs = combinations(unique_courses, 2)
        self.pairs = self.pairs.union(course_pairs)


def create_course_compatibility_matrix(fid_df: pd.DataFrame, teacher_df: pd.DataFrame):
    fid_df['id'] = fid_df['pid'] + " - " + fid_df['fid'].astype(str)
    fid_df['level'] = fid_df['lcid'].str[10]

    course_compatibilities = []    

    # Required courses in the same field and same year cannot be taught in same slot:
    mandatory_in_field = fid_df[fid_df['year'].ne(-1) & fid_df['category'].eq('M')].groupby('id').lcid.apply(list)
    mandatory = Compatibility(
        can_share_slot='no',
        reason='mandatory_within_field_year',
        data = mandatory_in_field.to_dict()
    )
    course_compatibilities.append(mandatory)

    # avoid any courses in thee same field and year
    in_same_field_year = fid_df.groupby(['id', 'year']).lcid.apply(list)
    data = in_same_field_year.to_dict()
    in_same_field_year_pairs = (
        pair 
        for courses in data.values()
        for pair in combinations(courses, 2)
    )
    field_year = Compatibility(can_share_slot='avoid', reason='in_same_field_year', data=data)
    field_year.pairs = set(in_same_field_year_pairs)
    course_compatibilities.append(field_year)

    # avoid any courses pair where one is -1 (any year) and the other is > 1
    data = ((f.pid, f.fid, f.year, f.lcid) for _, f in fid_df.iterrows() if f.year != 1)
    split = defaultdict(lambda: [[], []])
    for pid, fid, year, lcid in data:
        key = f'{pid} - {fid}'
        part = 0 if year == -1 else 1
        split[key][part] = lcid

    any_year_pairs = set((
        pair
        for fid, (any_year_courses, year_courses) in split.items()
        for pair in product(any_year_courses, year_courses)
    ))
    any_year = Compatibility(can_share_slot='avoid', reason='any_year', data={})
    any_year.data = data  # type: ignore
    any_year.pairs = any_year_pairs
    course_compatibilities.append(any_year)

    # F and M courses within the same field cannot be cotaught
    masters_in_field = fid_df[fid_df['level'].isin(['M', 'F'])].groupby('id').lcid.apply(list)
    masters = Compatibility(
        can_share_slot='no',
        reason='masters_within_field',
        data=masters_in_field.to_dict()
    )
    course_compatibilities.append(masters)

    # teachers cannot teach two slots at the same time
    teachers_courses = teacher_df[teacher_df['include']].groupby('id')['cid'].apply(list)
    teachers = Compatibility(
        can_share_slot='no',
        reason='taught_by_same_teacher',
        data=teachers_courses.to_dict()
    )
    course_compatibilities.append(teachers)

    return course_compatibilities

 
