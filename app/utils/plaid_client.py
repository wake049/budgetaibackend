import os
from dotenv import load_dotenv
from plaid.api import plaid_api
from plaid.configuration import Configuration
from plaid.api_client import ApiClient

load_dotenv()

plaid_client_id = os.getenv("PLAID_CLIENT_ID")
plaid_secret = os.getenv("PLAID_SECRET")
plaid_env = os.getenv("PLAID_ENV", "sandbox")

env_map = {
    "sandbox": "https://sandbox.plaid.com",
    "development": "https://development.plaid.com",
    "production": "https://production.plaid.com"
}

if not plaid_client_id or not plaid_secret:
    print("⚠️ Plaid credentials not found in environment variables")
    plaid_client = None
else:
    try:
        configuration = Configuration(
            host=env_map.get(plaid_env, "https://sandbox.plaid.com"),
            api_key={
                "clientId": plaid_client_id,
                "secret": plaid_secret
            }
        )
        api_client = ApiClient(configuration)
        plaid_client = plaid_api.PlaidApi(api_client)
        print(f"✅ Plaid client initialized for {plaid_env} environment")
    except Exception as e:
        print(f"❌ Failed to initialize Plaid client: {e}")
        plaid_client = None