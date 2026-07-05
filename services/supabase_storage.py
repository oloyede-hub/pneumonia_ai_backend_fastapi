import os

# I will test to remove you later
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL is not set in the environment variables.")
if not SUPABASE_SECRET_KEY:
    raise RuntimeError("SUPABASE_SECRET_KEY is not set in the environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)


def check_connection():
      try:
            buckets = supabase.storage.from_("uploads")
            print("Connection successful. Buckets:", buckets)
            return True
      except Exception as e:
        print("Connection failed:", str(e))
        return False

