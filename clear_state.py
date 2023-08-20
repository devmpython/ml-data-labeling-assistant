import streamlit as st
from enum import Enum


class DirtyState(Enum):
    NOT_DIRTY = "NOT_DIRTY"
    DIRTY = "DIRTY"
    UNHANDLED_SUBMIT = "UNHANDLED_SUBMIT"


class StreamlitHelper:
    @staticmethod
    def get_current_dirty_state() -> DirtyState:
        """
        Returns the current dirty state from the session state.
        Default is DirtyState.NOT_DIRTY if not set.
        """
        return DirtyState(st.session_state.get("dirty_state", DirtyState.NOT_DIRTY.value))

    @staticmethod
    def set_dirty_state(state: 'DirtyState') -> None:
        """
        Set the given dirty state in the session state.
        """
        st.session_state["dirty_state"] = state.value

    @staticmethod
    def clear_container(submit_clicked: bool) -> bool:
        """
        Determines if the container should be cleared based on the current dirty state
        and the submit action.
        """
        current_state = StreamlitHelper.get_current_dirty_state()

        if current_state == DirtyState.DIRTY:
            next_state = DirtyState.UNHANDLED_SUBMIT if submit_clicked else DirtyState.NOT_DIRTY
            StreamlitHelper.set_dirty_state(next_state)
            if next_state == DirtyState.UNHANDLED_SUBMIT:
                st.experimental_rerun()
            return False

        if submit_clicked or current_state == DirtyState.UNHANDLED_SUBMIT:
            StreamlitHelper.set_dirty_state(DirtyState.DIRTY)
            return True

        return False
