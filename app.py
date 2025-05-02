# app.py
import streamlit as st
import numpy as np
import pandas as pd
import os
import face_recognition
import pennylane as qml
from PIL import Image
import datetime
import time

# Page setup
st.set_page_config(page_title="Quantum Identity Verification", layout="centered")
st.markdown("<h1 style='text-align: center;'>🔐 Quantum Identity Verification</h1>", unsafe_allow_html=True)

# Ensure user database exists
if not os.path.exists("users.csv"):
    pd.DataFrame(columns=["first_name", "last_name", "file", "raw_file", "timestamp"]).to_csv("users.csv", index=False)

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = "home"
if "last_interaction" not in st.session_state:
    st.session_state.last_interaction = time.time()

# Reset function
def reset():
    st.session_state.clear()
    st.session_state.step = "home"
    st.rerun()

# Idle timeout (3 minutes)
def check_idle_timeout():
    if time.time() - st.session_state.last_interaction > 180:
        reset()

# Capture face and produce quantum + raw vectors
def capture_face_and_encode(image):
    st.session_state.last_interaction = time.time()
    img_np = np.array(image)
    locs = face_recognition.face_locations(img_np)
    if not locs:
        return None, None, "No face detected. Please try again."
    encs = face_recognition.face_encodings(img_np, locs)
    if not encs:
        return None, None, "Face encoding failed."
    raw_vec = encs[0]
    norm_vec = raw_vec / np.linalg.norm(raw_vec)
    reduced = norm_vec[:4] * np.pi
    dev = qml.device("default.qubit", wires=len(reduced))

    @qml.qnode(dev)
    def circuit(x):
        for i in range(len(reduced)):
            qml.RY(x[i], wires=i)
        return qml.state()

    return circuit(reduced), raw_vec, None

# Home page
if st.session_state.step == "home":
    st.subheader("Welcome! Choose an action and click Continue.")
    action = st.radio("Action:", ["Register", "Verify", "Delete Record"], key="home_action")
    if st.button("Continue", key="home_continue"):
        st.session_state.last_interaction = time.time()
        st.session_state.step = action.lower().replace(" ", "_")
        st.rerun()

# Registration flow - name
elif st.session_state.step == "register":
    check_idle_timeout()
    st.subheader("👤 Register Identity")
    fn = st.text_input("First Name", key="reg_fn")
    ln = st.text_input("Last Name", key="reg_ln")
    if st.button("Continue", key="reg_name_continue"):
        if not fn.strip() or not ln.strip():
            st.error("All fields are required.")
        else:
            st.session_state.fname = fn.strip()
            st.session_state.lname = ln.strip()
            st.session_state.step = "register_photo"
            st.session_state.last_interaction = time.time()
            st.rerun()

# Registration flow - photo
elif st.session_state.step == "register_photo":
    check_idle_timeout()
    st.subheader(f"📷 Capture Face for {st.session_state.fname} {st.session_state.lname}")
    photo = st.camera_input("Take a photo", key="reg_camera")
    if "registered" not in st.session_state:
        st.session_state.registered = False
    if photo and not st.session_state.registered and st.button("Finish Registration", key="finish_reg"):
        qstate, raw_vec, err = capture_face_and_encode(Image.open(photo))
        if err:
            st.error(err)
        else:
            df = pd.read_csv("users.csv")
            name_full = f"{st.session_state.fname} {st.session_state.lname}"
            # Check duplicates
            for _, row in df.iterrows():
                saved_raw = row["raw_file"]
                if os.path.exists(saved_raw):
                    saved_vec = np.load(saved_raw)
                    dist = face_recognition.face_distance([saved_vec], raw_vec)[0]
                    if dist < 0.4:
                        stored_name = f"{row['first_name']} {row['last_name']}"
                        if stored_name == name_full:
                            st.warning("⚠️ You are already registered with this name and face.")
                        else:
                            st.error("❌ This face has already been registered under a different name.")
                        st.session_state.registered = False
                        st.button("Go back home", key="reg_error_home", on_click=reset)
                        st.stop()
            # Save new user
            file_np = f"{name_full.replace(' ', '_')}.npy"
            raw_np = f"{name_full.replace(' ', '_')}_raw.npy"
            np.save(file_np, qstate)
            np.save(raw_np, raw_vec)
            df = df[df.file != file_np]
            df.loc[len(df)] = [st.session_state.fname, st.session_state.lname, file_np, raw_np, datetime.datetime.now()]
            df.to_csv("users.csv", index=False)
            st.session_state.registered = True
            st.success("✅ Registration complete")
    if st.session_state.registered:
        if st.button("Go back home", key="reg_success_home"):
            reset()

# Verification flow - name
elif st.session_state.step == "verify":
    check_idle_timeout()
    st.subheader("🔎 Verify Identity")
    fn2 = st.text_input("First Name", key="ver_fn")
    ln2 = st.text_input("Last Name", key="ver_ln")
    if st.button("Continue", key="ver_name_continue"):
        if not fn2.strip() or not ln2.strip():
            st.error("All fields are required.")
        else:
            df2 = pd.read_csv("users.csv")
            rec = df2[(df2.first_name == fn2.strip()) & (df2.last_name == ln2.strip())]
            if rec.empty:
                st.error("No registration found with that name.")
            else:
                st.session_state.fname = fn2.strip()
                st.session_state.lname = ln2.strip()
                st.session_state.step = "verify_photo"
                st.session_state.last_interaction = time.time()
                st.rerun()

# Verification flow - photo
elif st.session_state.step == "verify_photo":
    check_idle_timeout()
    st.subheader(f"📸 Face Verification for {st.session_state.fname} {st.session_state.lname}")
    photo2 = st.camera_input("Capture your face", key="ver_camera")
    if "verified" not in st.session_state:
        st.session_state.verified = False
    if photo2 and not st.session_state.verified and st.button("Verify", key="verify_btn"):
        q2, _, err2 = capture_face_and_encode(Image.open(photo2))
        if err2:
            st.error(err2)
        else:
            ref_file = f"{st.session_state.fname}_{st.session_state.lname}.npy"
            if not os.path.exists(ref_file):
                st.error("User data file not found. Please register again.")
            else:
                ref = np.load(ref_file)
                fid = np.abs(np.vdot(ref, q2)) ** 2
                st.metric("Fidelity Score", f"{fid:.6f}")
                if fid >= 0.95:
                    st.success("✅ Identity Verified")
                else:
                    st.error("❌ Identity could not be verified")
            st.session_state.verified = True
    if st.session_state.verified:
        if st.button("Go back home", key="ver_success_home"):
            reset()

# Delete record flow - name
elif st.session_state.step == "delete_record":
    check_idle_timeout()
    st.subheader("🗑️ Delete Record")
    fn3 = st.text_input("First Name", key="del_fn")
    ln3 = st.text_input("Last Name", key="del_ln")
    if st.button("Continue", key="del_name_continue"):
        if not fn3.strip() or not ln3.strip():
            st.error("All fields are required.")
        else:
            df3 = pd.read_csv("users.csv")
            rec3 = df3[(df3.first_name == fn3.strip()) & (df3.last_name == ln3.strip())]
            if rec3.empty:
                st.error("No registration found with that name.")
            else:
                st.session_state.fname = fn3.strip()
                st.session_state.lname = ln3.strip()
                st.session_state.step = "delete_photo"
                st.session_state.last_interaction = time.time()
                st.rerun()

# Delete record flow - photo
elif st.session_state.step == "delete_photo":
    check_idle_timeout()
    st.subheader(f"📸 Confirm Deletion for {st.session_state.fname} {st.session_state.lname}")
    photo3 = st.camera_input("Capture your face", key="del_camera")
    if "deleted" not in st.session_state:
        st.session_state.deleted = False
    if photo3 and not st.session_state.deleted and st.button("Delete Record", key="delete_btn"):
        q3, raw3, err3 = capture_face_and_encode(Image.open(photo3))
        if err3:
            st.error(err3)
        else:
            file3 = f"{st.session_state.fname}_{st.session_state.lname}.npy"
            raw3_file = f"{st.session_state.fname}_{st.session_state.lname}_raw.npy"
            if not os.path.exists(file3) or not os.path.exists(raw3_file):
                st.error("User data file not found. Please register first.")
            else:
                saved_vec3 = np.load(raw3_file)
                dist3 = face_recognition.face_distance([saved_vec3], raw3)[0]
                if dist3 < 0.4:
                    # Delete files and CSV entry
                    os.remove(file3)
                    os.remove(raw3_file)
                    df4 = pd.read_csv("users.csv")
                    df4 = df4[~((df4.first_name == st.session_state.fname) & (df4.last_name == st.session_state.lname))]
                    df4.to_csv("users.csv", index=False)
                    st.session_state.deleted = True
                    st.success("✅ Record deleted successfully")
                else:
                    st.error("❌ Provided information does not match our records.")
    if st.session_state.deleted:
        if st.button("Go back home", key="del_success_home"):
            reset()
