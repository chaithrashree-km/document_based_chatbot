import random
import time
from locust import HttpUser, task, between, events

SHORT_QUESTIONS = [
    "What is this document about?",
    "What is the main topic?",
    "Give a one line summary.",
    "What does this file contain?",
    "Describe the document briefly.",
]

TEST_USERS = [
    {"email": f"loadtest_user_{i}@test.com", "password": "LoadTest@123"}
    for i in range(1, 151)
]

response_times = []

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    if exception is None and name == "/chat":
        response_times.append(response_time)

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    if response_times:
        avg = sum(response_times) / len(response_times)
        p95 = sorted(response_times)[int(len(response_times) * 0.95)]
        p99 = sorted(response_times)[int(len(response_times) * 0.99)]
        print(f"\n{'='*50}")
        print(f"  /chat endpoint custom summary")
        print(f"  Total requests : {len(response_times)}")
        print(f"  Avg latency    : {avg:.0f} ms")
        print(f"  P95 latency    : {p95:.0f} ms")
        print(f"  P99 latency    : {p99:.0f} ms")
        print(f"{'='*50}\n")


class ChatUser(HttpUser):

    weight = 7
    wait_time = between(5, 8)  

    def on_start(self):
        self.token = None
        self.questions_asked = 0     
        self.user_data = random.choice(TEST_USERS)
        self._login()

    def _login(self):
        with self.client.post(
            "/login",
            json=self.user_data,
            catch_response=True,
            name="/login"
        ) as resp:
            if resp.status_code == 200:
                self.token = resp.json().get("access_token")
                resp.success()
            elif resp.status_code == 404:
                self._signup()
            else:
                resp.failure(f"Login failed: {resp.status_code} {resp.text}")

    def _signup(self):
        payload = {
            "username": self.user_data["email"].split("@")[0],
            "email": self.user_data["email"],
            "password": self.user_data["password"],
        }
        with self.client.post("/signup", json=payload, catch_response=True, name="/signup") as resp:
            if resp.status_code == 200:
                self.token = resp.json().get("access_token")
                resp.success()
            else:
                resp.failure(f"Signup failed: {resp.status_code}")

    def _auth_headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    # @task(10)
    # def ask_question(self):
    #     if not self.token:
    #         self._login()
    #         return

    #     if self.questions_asked >= 2:   
    #         return

    #     question = random.choice(SHORT_QUESTIONS)  

    #     with self.client.post(
    #         "/chat",
    #         json={"question": question},
    #         headers=self._auth_headers(),
    #         catch_response=True,
    #         name="/chat",
    #         timeout=120
    #     ) as resp:
    #         if resp.status_code == 200:
    #             data = resp.json()
    #             if "Response" not in data and "response" not in data:
    #                 resp.failure("Response key missing in chat reply")
    #             else:
    #                 self.questions_asked += 1  
    #                 resp.success()
    #         elif resp.status_code == 429:
    #             resp.failure("Rate limited (429)")
    #         elif resp.status_code == 401:
    #             self._login()
    #             resp.failure("Token expired — re-logged in")
    #         else:
    #             resp.failure(f"Chat failed: {resp.status_code}")

    @task(1)
    def new_chat_session(self):
        if not self.token:
            return
        if self.questions_asked >= 2:  
            return
        with self.client.post(
            "/new_chat",
            headers=self._auth_headers(),
            catch_response=True,
            name="/new_chat"
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"new_chat failed: {resp.status_code}")

    @task(1)
    def get_chat_history(self):
        if not self.token:
            return
        with self.client.get(
            "/get_chats_by_user",
            headers=self._auth_headers(),
            catch_response=True,
            name="/get_chats_by_user"
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"get_chats failed: {resp.status_code}")


class UploadUser(HttpUser):
    """
    Simulates users who upload documents.
    Weight=2 → 20% of virtual users.
    UNCHANGED from original.
    """
    weight = 2
    wait_time = between(15, 45)

    def on_start(self):
        self.token = None
        self.user_data = random.choice(TEST_USERS)
        self._login()

    def _login(self):
        with self.client.post("/login", json=self.user_data,
                              catch_response=True, name="/login") as resp:
            if resp.status_code == 200:
                self.token = resp.json().get("access_token")
            elif resp.status_code == 404:
                payload = {
                    "username": self.user_data["email"].split("@")[0],
                    "email": self.user_data["email"],
                    "password": self.user_data["password"],
                }
                r = self.client.post("/signup", json=payload)
                if r.status_code == 200:
                    self.token = r.json().get("access_token")

    @task
    def upload_document(self):
        if not self.token:
            return

        fake_content = (
            f"This is a test document uploaded at {time.time()}. "
            "It contains information about AI, machine learning, and document retrieval."
        ).encode("utf-8")

        with self.client.post(
            "/upload",
            files={"file": ("test_doc.txt", fake_content, "text/plain")},
            headers={"Authorization": f"Bearer {self.token}"},
            catch_response=True,
            name="/upload",
            timeout=60
        ) as resp:
            if resp.status_code == 200:
                resp.success()
                task_id = resp.json().get("task_id")
                if task_id:
                    self._poll_task(task_id)
            elif resp.status_code == 429:
                resp.failure("Upload rate limited")
            else:
                resp.failure(f"Upload failed: {resp.status_code}")

    def _poll_task(self, task_id: str):
        for _ in range(5):
            time.sleep(3)
            r = self.client.get(
                f"/upload/status/{task_id}",
                headers={"Authorization": f"Bearer {self.token}"},
                name="/upload/status"
            )
            if r.status_code == 200 and r.json().get("status") in ("SUCCESS", "FAILURE"):
                break


class HealthCheckUser(HttpUser):
    """
    Simulates lightweight monitoring probes.
    Weight=1 → 10% of virtual users.
    UNCHANGED from original.
    """
    weight = 1
    wait_time = between(1, 3)

    @task
    def health(self):
        self.client.get("/health", name="/health")