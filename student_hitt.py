# type: ignore
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel
from streamlit_calendar import calendar

from load_data import load, load_enriched_schedule


class TimeSlot(NamedTuple):
    """Represents a time slot with day and time information."""

    weekday: int
    start_time: str
    end_time: str
    location: str
    is_lecture: bool = True


class Course(BaseModel):
    """Model for a course with validation."""

    lcid: str
    cid: str
    name: str
    teacher: str
    location: str
    building: Optional[str] = None
    start_week: Optional[int] = None
    end_week: Optional[int] = None
    lecture: bool = True
    sidcount: Optional[int] = None
    level: Optional[str] = None
    season: Optional[str] = None
    nameeng: Optional[str] = None
    timeslots: List[TimeSlot] = []


class TimetableManager:
    """Manages the timetable data and operations."""

    def __init__(self):
        """Initialize the timetable manager with reference dates and data."""
        # Reference start week for calendar
        self.reference_monday: datetime = datetime(2025, 2, 17)
        self.load_data()

    def load_data(self) -> None:
        """Load the schedule and related data."""
        # Load raw data
        (
            self.room_df,
            self.fid_df,
            self.clash_df,
            self.name_df,
            self.schedule_df
        ) = load()

        # Build course lookup
        self.all_courses_df = self.schedule_df.drop_duplicates(
            subset=["lcid", "name", "weekday", "start", "end", "location"]
        )
        self.all_course_lookup: Dict[str, str] = dict(
            zip(self.all_courses_df["lcid"], self.all_courses_df["name"])
        )

        # Prepare a lookup for student clashes
        self.clash_lookup: Dict[Tuple[str, str], int] = {
            (row["course1"], row["course2"]): row["n_students"]
            for _, row in self.clash_df.iterrows()
        }
        # Make clash lookup symmetric
        self.clash_lookup.update({(b, a): v for (a, b), v in self.clash_lookup.items()})

    def filter_schedule(self, include_tutorials: bool) -> pd.DataFrame:
        """Filter schedule based on tutorial inclusion preference."""
        if not include_tutorials:
            return self.schedule_df[self.schedule_df["lecture"]]
        return self.schedule_df

    def get_program_schedule(
        self, program_id: str, year: Union[int, float]
    ) -> pd.DataFrame:
        """Get schedule for a specific program and year."""
        schedule_with_program = pd.merge(
            self.schedule_df,
            self.fid_df[["lcid", "pid", "year"]],
            on="lcid",
            how="left",
        )

        # Filter relevant courses from selected program/year (+ year-independent courses)
        return schedule_with_program[
            (schedule_with_program["pid"] == program_id)
            & (
                (schedule_with_program["year"] == year)
                | (schedule_with_program["year"] == -1)
            )
        ].drop_duplicates(subset=["lcid", "weekday", "start", "end"])

    def get_study_programs(self) -> List[str]:
        """Get list of available study programs."""
        schedule_with_program = pd.merge(
            self.schedule_df,
            self.fid_df[["lcid", "pid", "year"]],
            on="lcid",
            how="left",
        )
        return sorted(schedule_with_program["pid"].dropna().unique())

    def get_available_years(self, program_id: str) -> List[Union[int, float]]:
        """Get available years for a specific program."""
        schedule_with_program = pd.merge(
            self.schedule_df,
            self.fid_df[["lcid", "pid", "year"]],
            on="lcid",
            how="left",
        )
        return sorted(
            schedule_with_program[schedule_with_program["pid"] == program_id]["year"]
            .dropna()
            .unique()
        )

    def parse_event_time(self, row: Any) -> Tuple[datetime, datetime]:
        """Parse event time from a row."""
        weekday = int(row.weekday)
        start_time = datetime.strptime(row.start, "%H:%M:%S").time()
        end_time = datetime.strptime(row.end, "%H:%M:%S").time()
        date = self.reference_monday + timedelta(days=weekday - 1)
        start = datetime.combine(date, start_time)
        end = datetime.combine(date, end_time)
        return start, end

    def check_conflicts(self, visible_schedule: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for scheduling conflicts between courses."""
        conflicts = []
        visible_courses = visible_schedule[
            ["lcid", "name", "weekday", "start", "end"]
        ].drop_duplicates()

        for course1, course2 in combinations(
            visible_courses.itertuples(index=False), 2
        ):
            if course1.weekday != course2.weekday:
                continue  # Only compare courses on same day

            start1, end1 = self.parse_event_time(course1)
            start2, end2 = self.parse_event_time(course2)

            if max(start1, start2) < min(end1, end2):  # They overlap
                clash_count = self.clash_lookup.get((course1.lcid, course2.lcid))
                if clash_count:
                    conflicts.append(
                        {
                            "Course 1": course1.name,
                            "Course 2": course2.name,
                            "Weekday": ["Mon", "Tue", "Wed", "Thu", "Fri"][
                                course1.weekday - 1
                            ],
                            "Time 1": f"{course1.start}–{course1.end}",
                            "Time 2": f"{course2.start}–{course2.end}",
                            "Students in Clash": clash_count,
                        }
                    )

        return conflicts

    def compute_minutes(self, start_str: str, end_str: str) -> float:
        """Compute minutes between two time strings."""
        start = datetime.strptime(start_str, "%H:%M:%S")
        end = datetime.strptime(end_str, "%H:%M:%S")
        return np.round((end - start).total_seconds() / 60 / 46)

    def compute_duration_summary(self, visible_schedule: pd.DataFrame) -> pd.DataFrame:
        """Compute duration summary for visible schedule."""
        # Compute duration per time slot
        visible_schedule["Units (40min)"] = visible_schedule.apply(
            lambda row: self.compute_minutes(row["start"], row["end"]), axis=1
        )

        # Group by course
        return (
            visible_schedule.groupby(["lcid", "name"], as_index=False)["Units (40min)"]
            .sum()
            .sort_values(by="Units (40min)", ascending=False)
        )

    def to_calendar_event(self, row: pd.Series) -> Dict[str, str]:
        """Convert a row to a calendar event."""
        weekday = int(row["weekday"])
        date = self.reference_monday + timedelta(days=weekday - 1)
        start_time = datetime.strptime(row["start"], "%H:%M:%S").time()
        end_time = datetime.strptime(row["end"], "%H:%M:%S").time()
        start = datetime.combine(date, start_time).isoformat()
        end = datetime.combine(date, end_time).isoformat()
        title = f"{row['name']} [{row['cid']}] - {row['teacher']} in {row['location']}"

        return {"id": row["lcid"], "title": title, "start": start, "end": end}


class TimetableApp:
    """Main Streamlit application class."""

    def __init__(self):
        """Initialize the application state."""
        self.manager = TimetableManager()
        self.initialize_session_state()

    def initialize_session_state(self) -> None:
        """Initialize session state variables."""
        if "selected_courses" not in st.session_state:
            st.session_state.selected_courses: Set[str] = set()

        if "custom_courses" not in st.session_state:
            st.session_state.custom_courses: List[Dict[str, Any]] = []

        if "last_selection" not in st.session_state:
            st.session_state.last_selection: Tuple[str, Union[int, float]] = ("", 0)

    def sidebar_settings(self) -> Tuple[bool, str, Union[int, float]]:
        """Render sidebar settings and return user selections."""
        with st.sidebar:
            st.title("Timetable Settings")

            # Include tutorials checkbox
            include_tutorials = st.checkbox("Include tutorials", value=False)

            # Study program selection
            default_program = "IÐN262"
            study_programs = self.manager.get_study_programs()

            selected_program = st.selectbox(
                "Choose a program of study (pid):",
                study_programs,
                index=(
                    study_programs.index(default_program)
                    if default_program in study_programs
                    else 0
                ),
            )

            # Year selection
            available_years = self.manager.get_available_years(selected_program)
            selected_year = st.selectbox("Choose study year:", available_years)

            # Reset everything when program or year changes
            if st.session_state.last_selection != (selected_program, selected_year):
                st.session_state.selected_courses = set()
                st.session_state.custom_courses = []
                st.session_state.last_selection = (selected_program, selected_year)

        return include_tutorials, selected_program, selected_year

    def display_download_button(
        self, selected_program: str, selected_year: int | float
    ) -> None:
        with st.sidebar:
            st.markdown("---")

            # Export button
            if st.button("📥 Download selected timetable"):
                visible_schedule = self.get_visible_schedule(
                    selected_program, selected_year
                )
                st.download_button(
                    label="Download CSV",
                    data=visible_schedule.to_csv(index=False),
                    file_name="my_timetable.csv",
                    mime="text/csv",
                )

    def get_visible_schedule(
        self, program_id: str, year: Union[int, float]
    ) -> pd.DataFrame:
        """Get visible schedule based on selected courses."""
        # Get program schedule
        program_schedule = self.manager.get_program_schedule(program_id, year)

        # Add custom courses from session
        if st.session_state.custom_courses:
            custom_df = pd.DataFrame(st.session_state.custom_courses)
            program_schedule = pd.concat(
                [program_schedule, custom_df], ignore_index=True
            )

        # Auto-select all program courses on first load
        if not st.session_state.selected_courses:
            st.session_state.selected_courses = set(program_schedule["lcid"])

        # Merge program + full school + custom into the full course universe
        full_schedule_df = pd.concat(
            [program_schedule, self.manager.all_courses_df], ignore_index=True
        )
        full_schedule_df = full_schedule_df.drop_duplicates(
            subset=["lcid", "weekday", "start", "end"]
        )

        # Filter visible schedule
        return full_schedule_df[
            full_schedule_df["lcid"].isin(st.session_state.selected_courses)
        ]

    def display_calendar(
        self, visible_schedule: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Display the calendar and return selection."""

        # Convert rows to calendar events
        calendar_events = [
            self.manager.to_calendar_event(row)
            for _, row in visible_schedule.iterrows()
        ]

        # Display calendar and get selection
        return calendar(
            events=calendar_events,
            options={
                "initialView": "timeGridWeek",
                "initialDate": self.manager.reference_monday.date().isoformat(),
                "editable": True,
                "eventDurationEditable": True,
                "eventResizableFromStart": True,
                "snapDuration": "00:50:00",
                "slotDuration": "00:50:00",
                "slotMinTime": "08:20:00",
                "slotMaxTime": "23:59:59",
                "slotLabelFormat": {
                    "hour": "numeric",
                    "minute": "2-digit",
                    "hour12": False,
                },
                "locale": "is",
                "firstDay": 1,
                "headerToolbar": {
                    "left": "",
                    "center": "",
                    "right": "",
                },
                "dayHeaderFormat": {"weekday": "short"},
                "allDaySlot": False,
                "height": "auto",
            },
            key=f"calendar_{st.session_state.last_selection[0]}_{st.session_state.last_selection[1]}_{len(calendar_events)}",
        )

    def display_course_details(self, selection: Dict[str, Any]) -> None:
        """Display details for a selected course."""
        if selection and "eventClick" in selection:
            selected_event = selection["eventClick"]["event"]
            selected_lcid = str(selected_event.get("id"))

            course_rows = self.manager.schedule_df[
                self.manager.schedule_df["lcid"] == selected_lcid
            ]

            st.markdown("### 📘 Course Details")

            if not course_rows.empty:
                row = course_rows.iloc[0]

                # Constructing the markdown block as a string
                course_info = f"""\
                    Course: {row['name']} ({row['cid']})
                    Teacher(s): {row['teacher']}
                    Location: {row['location']} - {row.get('building', '')}
                    Scheduled weeks: {row.get('start_week', '?')}-{row.get('end_week', '?')}
                    Lecture? {'Yes' if row.get('lecture', True) else 'No'}
                    Students: {row.get('sidcount', 'N/A')}
                    Level / Season: {row.get('level', '')} / {row.get('season', '')}
                    English Name: {row.get('nameeng', '')}

                    Timeslots:
                    """

                for _, r in course_rows.iterrows():
                    day = ["Mon", "Tue", "Wed", "Thu", "Fri"][int(r["weekday"]) - 1]
                    start = r["start"]
                    end = r["end"]
                    loc = r["location"]
                    is_lecture = r.get("lecture", True)
                    label = "Lecture" if is_lecture else "Tutorial"
                    course_info += f"- {day}: {start}-{end} in {loc} {label}\n"

                # Editable textbox for the markdown content
                edited_text = st.text_area("Course Information", course_info, height=300)


            else:
                st.warning(f"No data found for selected course ID: {selected_lcid}")


    def display_course_management(self) -> None:
        """Display course management UI in the sidebar."""
        with st.sidebar:
            st.markdown("---")
            st.subheader("🎓 Customize Your Calendar")

            # Add course from full school course list (one at a time)
            add_course_lcids = [
                lcid
                for lcid in self.manager.all_course_lookup
                if lcid not in st.session_state.selected_courses
            ]
            selected_add = st.selectbox(
                "Add a course from the school:",
                options=[""] + sorted(add_course_lcids),
                format_func=lambda lcid: (
                    self.manager.all_course_lookup.get(lcid, "")
                    if lcid
                    else "Select a course..."
                ),
            )
            if selected_add:
                st.session_state.selected_courses.add(selected_add)
                st.success(f"Added: {self.manager.all_course_lookup.get(selected_add)}")
                st.rerun()

            # Remove selected courses
            if st.session_state.selected_courses:
                remove_lcid = st.selectbox(
                    "Remove a course from your calendar:",
                    options=[""] + sorted(st.session_state.selected_courses),
                    format_func=lambda lcid: (
                        self.manager.all_course_lookup.get(lcid, lcid)
                        if lcid
                        else "Select a course to remove..."
                    ),
                )
                if remove_lcid:
                    st.session_state.selected_courses.discard(remove_lcid)
                    st.warning(
                        f"Removed: {self.manager.all_course_lookup.get(remove_lcid, remove_lcid)}"
                    )
                    st.rerun()

    def display_conflicts(self, visible_schedule: pd.DataFrame) -> None:
        """Display scheduling conflicts."""
        st.markdown("---")
        st.subheader("⚠️ Scheduling Conflicts")

        conflicts = self.manager.check_conflicts(visible_schedule)

        # Show results
        if conflicts:
            st.warning(f"🚨 {len(conflicts)} course conflicts found")
            df_conflicts = pd.DataFrame(conflicts)
            st.dataframe(df_conflicts)
            total_students = sum(conf["Students in Clash"] for conf in conflicts)
            st.markdown(f"**👥 Total students in conflict: {total_students}**")
        else:
            st.success("✅ No clashes found among currently selected courses.")

    def display_duration_summary(self, visible_schedule: pd.DataFrame) -> None:
        """Display course duration summary."""
        st.markdown("---")
        st.subheader("🕒 Course Duration Summary")

        duration_summary = self.manager.compute_duration_summary(visible_schedule)

        # Display summary table
        st.dataframe(duration_summary, use_container_width=True)

        # Show total assigned minutes across all selected courses
        total_minutes = duration_summary["Units (40min)"].sum()
        st.markdown(
            f"**📊 Total sessions scheduled across all selected courses: {round(total_minutes)} sessions**"
        )

    def run(self) -> None:
        """Run the Streamlit application."""
        st.set_page_config(page_title="Academic Timetable", layout="wide")
        st.title("Stundatafla")

        # Get settings from sidebar
        include_tutorials, selected_program, selected_year = self.sidebar_settings()

        # Display course management UI in the sidebar
        self.display_course_management()

        # Display download button
        self.display_download_button(selected_program, selected_year)

        # Filter schedule based on tutorial inclusion
        filtered_schedule = self.manager.filter_schedule(include_tutorials)

        # Use the filtered schedule for this session
        self.manager.schedule_df = filtered_schedule

        # Rebuild all_courses_df based on the filtered schedule
        self.manager.all_courses_df = filtered_schedule.drop_duplicates(
            subset=["lcid", "name", "weekday", "start", "end", "location"]
        )
        self.manager.all_course_lookup = dict(
            zip(
                self.manager.all_courses_df["lcid"], self.manager.all_courses_df["name"]
            )
        )
        # Get visible schedule
        visible_schedule = self.get_visible_schedule(selected_program, selected_year)

        # Display calendar
        selection = self.display_calendar(visible_schedule)

        # Display course details if selected
        self.display_course_details(selection)

        # Display conflicts
        self.display_conflicts(visible_schedule)

        # Display duration summary
        self.display_duration_summary(visible_schedule)


if __name__ == "__main__":
    app = TimetableApp()
    app.run()
