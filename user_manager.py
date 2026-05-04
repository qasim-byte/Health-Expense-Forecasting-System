import json
import os
import hashlib
from datetime import datetime

class UserManager:
    def __init__(self, data_file="users.json"):
        self.data_file = data_file
        self.users = self.load_users()

    def load_users(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return {}

    def save_users(self):
        with open(self.data_file, 'w') as f:
            json.dump(self.users, f, indent=4)

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def signup(self, patient_id, password, name):
        if patient_id in self.users:
            return False, "Patient ID already exists"

        self.users[patient_id] = {
            "password": self.hash_password(password),
            "name": name,
            "reports": [],
            "created_at": datetime.now().isoformat()
        }
        self.save_users()
        return True, "Signup successful"

    def login(self, patient_id, password):
        if patient_id not in self.users:
            return False, "Patient ID not found"

        if self.users[patient_id]["password"] != self.hash_password(password):
            return False, "Incorrect password"

        return True, self.users[patient_id]["name"]

    def save_report(self, patient_id, report_data):
        if patient_id not in self.users:
            return False

        report_entry = {
            "timestamp": datetime.now().isoformat(),
            "data": report_data
        }
        self.users[patient_id]["reports"].append(report_entry)
        self.save_users()
        return True

    def get_reports(self, patient_id):
        if patient_id not in self.users:
            return []
        return self.users[patient_id]["reports"]

    def get_user_name(self, patient_id):
        if patient_id not in self.users:
            return None
        return self.users[patient_id]["name"]

    def update_user_info(self, patient_id, name):
        if patient_id not in self.users:
            return False
        self.users[patient_id]["name"] = name
        self.save_users()
        return True