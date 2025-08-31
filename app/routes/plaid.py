from urllib import response
from plaid.api import plaid_api
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.products import Products
from plaid.model.country_code import CountryCode
from plaid.model.link_token_account_filters import LinkTokenAccountFilters
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.configuration import Configuration
from plaid.model.link_token_create_request_auth import LinkTokenCreateRequestAuth
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.transactions_get_response import TransactionsGetResponse
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from plaid.model.accounts_get_request import AccountsGetRequest

from plaid import ApiClient
from dotenv import load_dotenv
import os
from fastapi import HTTPException, Request, APIRouter, Query
import uuid
from datetime import date, timedelta, datetime
import traceback

router = APIRouter()

load_dotenv()

configuration = Configuration(
    host="https://sandbox.plaid.com",
    api_key={
        'clientId': os.getenv("PLAID_CLIENT_ID"),
        'secret': os.getenv("PLAID_SECRET")
    }
)

api_client = ApiClient(configuration)
plaid_client = plaid_api.PlaidApi(api_client)

@router.post("/plaid/create_link_token")
async def create_link_token():
    try:
        request = LinkTokenCreateRequest(
            user=LinkTokenCreateRequestUser(client_user_id=str(uuid.uuid4())),
            client_name="BudgetAI",
            products=[Products("transactions")],
            country_codes=[CountryCode("US")],
            language="en"
        )

        response = plaid_client.link_token_create(request)
        # Extract the link_token from the response safely
        link_token = response.get('link_token') or response.link_token
        return {"link_token": link_token}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plaid/exchange_public_token")
async def exchange_public_token(request: Request):
    try:
        data = await request.json()
        public_token = data.get("public_token")
        if not public_token:
            raise HTTPException(status_code=400, detail="Missing public_token")

        exchange_request = ItemPublicTokenExchangeRequest(public_token=public_token)
        exchange_response = plaid_client.item_public_token_exchange(exchange_request)

        # Extract values safely to avoid circular references
        access_token = exchange_response.get('access_token') or exchange_response.access_token
        item_id = exchange_response.get('item_id') or exchange_response.item_id

        return {
            "access_token": access_token,
            "item_id": item_id
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/plaid/transactions")
async def get_plaid_transactions(access_token: str = Query(...)):
    try:
        start_date = date.today() - timedelta(days=30)
        end_date = date.today()

        options = TransactionsGetRequestOptions(count=100)
        request = TransactionsGetRequest(
            access_token=access_token,
            start_date=start_date,
            end_date=end_date,
            options=options
        )

        response: TransactionsGetResponse = plaid_client.transactions_get(request)
        # Extract transactions safely to avoid circular references
        transactions = response.get('transactions') or response.transactions

        formatted = format_plaid_transactions(transactions)
        return {"transactions": formatted}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/plaid/get_accounts")
async def get_accounts(request: Request):
    try:
        data = await request.json()
        access_token = data.get("access_token")

        req = AccountsGetRequest(access_token=access_token)
        response = plaid_client.accounts_get(req)
        accounts_raw = response.to_dict()["accounts"]

        accounts = []
        for acc in accounts_raw:
            accounts.append({
                "account_id": acc.get("account_id"),
                "name": acc.get("name"),
                "type": acc.get("type"),
                "subtype": acc.get("subtype"),
                "balance": {
                    "available": acc.get("balances", {}).get("available"),
                    "current": acc.get("balances", {}).get("current")
                }
            })

        return {"accounts": accounts}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Plaid error: " + str(e))



def format_plaid_transactions(plaid_txns):
    formatted = []
    for txn in plaid_txns:
        # Safely extract transaction data to avoid circular references
        amount = txn.get('amount') or txn.amount
        category = txn.get('category') or txn.category
        name = txn.get('name') or txn.name
        date = txn.get('date') or txn.date
        
        formatted.append({
            "amount": amount,
            "category": category[0] if category and len(category) > 0 else "Uncategorized",
            "description": name,
            "timestamp": date.isoformat() if hasattr(date, 'isoformat') else str(date)
        })
    return formatted