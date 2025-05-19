"""
Versão corrigida do bot MetaTrader5 LLM - Compatível com LangChain
"""
from dotenv import load_dotenv
import os

load_dotenv()



import os
import json
import time
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import queue
import re
import warnings
from prompts import initial_context_prompt

print(f"🔑 API KEY: {os.getenv('OPENAI_API_KEY')[:20]}... (parcial)")


# Desativar avisos de depreciação
warnings.filterwarnings("ignore")
# Atualizar imports
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from indicators import calculate_atr, calculate_directional_entropy, calculate_ema
from utils import parse_llm_response, format_trade_for_feedback, calculate_position_performance, detect_price_patterns
# Adicione este código no início do arquivo app.py, após as importações
# Add this with other global variables near the top
initial_market_context = None
use_initial_context_enabled = False
# Custom CSS para melhorar a aparência da interface
app_css = """
/* Chat styles */
.chat-container {
    scrollbar-width: thin;
    scrollbar-color: #6c757d #343a40;
}

.chat-container::-webkit-scrollbar {
    width: 6px;
}

.chat-container::-webkit-scrollbar-track {
    background: #343a40;
}

.chat-container::-webkit-scrollbar-thumb {
    background-color: #6c757d;
    border-radius: 3px;
}

.user-message, .assistant-message {
    max-width: 85%;
    padding: 8px 12px;
    margin-bottom: 10px;
    border-radius: 12px;
    word-wrap: break-word;
}

.user-message {
    background-color: #0d6efd;
    color: white;
    align-self: flex-end;
    border-bottom-right-radius: 2px;
}

.assistant-message {
    background-color: #2a2e32;
    color: #f8f9fa;
    align-self: flex-start;
    border-bottom-left-radius: 2px;
}

.timestamp {
    font-size: 0.7rem;
    color: rgba(255, 255, 255, 0.5);
    margin-top: 2px;
    text-align: right;
}

.suggestion-btn {
    transition: all 0.2s;
    border-radius: 15px;
    font-size: 0.8rem;
}

.suggestion-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
}

.chat-card {
    border-left: 4px solid #0d6efd;
}

.historical-date-marker {
    text-align: center;
    color: #adb5bd;
    font-size: 0.8rem;
    margin: 15px 0;
    position: relative;
}

.historical-date-marker::before, 
.historical-date-marker::after {
    content: "";
    display: inline-block;
    width: 25%;
    height: 1px;
    background-color: rgba(173, 181, 189, 0.5);
    vertical-align: middle;
    margin: 0 10px;
}

.timeframe-selector {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin: 10px 0;
}

.timeframe-btn {
    font-size: 0.8rem;
    padding: 3px 8px;
    border-radius: 12px;
}

.animate-pulse {
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% {
        opacity: 1;
    }
    50% {
        opacity: 0.5;
    }
    100% {
        opacity: 1;
    }
}

/* Melhorias estéticas para o dashboard */
.reasoning-text p {
    line-height: 1.5;
    margin-bottom: 0.8rem;
}

/* Gradiente para a barra de confiança */
.confidence-gradient {
    background: linear-gradient(to right, #dc3545, #ffc107, #0dcaf0, #0d6efd, #198754);
    height: 6px;
    width: 100%;
    margin-top: -6px;
    border-radius: 0 0 0.25rem 0.25rem;
}

/* Estilos para os cards de fatores */
.factor-card {
    transition: transform 0.2s;
}
.factor-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* Estilo para o botão de enviar */
#send-button {
    transition: all 0.2s ease;
}
#send-button:hover {
    transform: translateX(2px);
    box-shadow: 0 0 10px rgba(13, 110, 253, 0.5);
}

/* Melhorias para o histórico de trades */
.trade-card {
    transition: all 0.2s;
}
.trade-card:hover {
    transform: scale(1.02);
}

/* Destaque para o header */
.app-header {
    background: linear-gradient(to right, #343a40, #495057);
    border-radius: 0.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

/* Melhorias para os cards */
.dashboard-card {
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: all 0.3s;
}
.dashboard-card:hover {
    box-shadow: 0 6px 10px rgba(0,0,0,0.15);
}

/* Estilo para valores de confiança */
.confidence-value {
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    display: inline-block;
    font-weight: bold;
}
.confidence-bullish {
    background-color: rgba(25, 135, 84, 0.1);
    color: #198754;
}
.confidence-bearish {
    background-color: rgba(220, 53, 69, 0.1);
    color: #dc3545;
}
.confidence-neutral {
    background-color: rgba(13, 202, 240, 0.1);
    color: #0dcaf0;
}
"""

# Adicione o CSS à aplicação
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)

app.title = "MetaTrader5 LLM risk management"

# Aplicar o CSS customizado
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
''' + app_css + '''
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
# Global variables
running = False
trade_queue = queue.Queue()
memory = None
llm_chain = None
current_position = 0  # Current number of contracts
max_contracts = 5  # Default max contracts
confidence_level = 0  # Market confidence (-100 to 100)
llm_reasoning = ""
total_pnl = 0.0
market_direction = "Neutral"  # Market direction (Bullish, Bearish, Neutral)
trade_history = []
support_resistance_levels = []

# Define the timeframe mapping
timeframe_dict = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

# Initialize the application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "MetaTrader5 LLM Trading Bot"

# Set API Key from environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Fix for the update_chat function to resolve the AttributeError

@app.callback(
    [Output("chat-messages", "children"),
     Output("chat-visual-results", "children"),
     Output("chat-visual-results", "style"),
     Output("chat-history", "data"),
     Output("chat-input", "value")],
    [Input("send-button", "n_clicks"),
     Input("chat-input", "n_submit"),
     Input("suggestion-1", "n_clicks"),
     Input("suggestion-2", "n_clicks"),
     Input("suggestion-3", "n_clicks"),
     Input("suggestion-4", "n_clicks")],
    [State("chat-input", "value"),
     State("chat-messages", "children"),
     State("chat-history", "data"),
     State("symbol-input", "value"),
     State("timeframe-dropdown", "value")]
)

def log_candle_data(symbol, timeframe, candle_data, analysis, file_path=None):
    """
    Log candle data, indicator values, and LLM analysis to an Excel file
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Timeframe code (e.g. "H4")
        candle_data (str): JSON string containing candle data
        analysis (dict): Dictionary containing LLM analysis
        file_path (str, optional): Path to save the Excel file. If None, a default path is used.
    """
    # Use default file path if none provided
    if file_path is None:
        file_path = f"metatradebot_log_{symbol}_{timeframe}.xlsx"
    
    # Parse candle data JSON
    try:
        candle_dict = json.loads(candle_data)
    except:
        print(f"Error parsing candle data JSON: {candle_data}")
        return
    
    # Create a data dictionary for this entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract OHLC data
    ohlc = candle_dict.get("ohlc", {})
    
    # Extract indicator values
    indicators = candle_dict.get("indicators", {})
    
    # Extract price action data
    price_action = candle_dict.get("price_action", {})
    
    # Combine all data into a single row
    data = {
        "timestamp": timestamp,
        "symbol": symbol,
        "timeframe": timeframe,
        "candle_time": candle_dict.get("timestamp", ""),
        
        # OHLC data
        "open": ohlc.get("open", 0),
        "high": ohlc.get("high", 0),
        "low": ohlc.get("low", 0),
        "close": ohlc.get("close", 0),
        "volume": candle_dict.get("volume", 0),
        
        # Indicator values
        "atr": indicators.get("atr", 0),
        "directional_entropy": indicators.get("directional_entropy", 0),
        "ema9": indicators.get("ema9", 0),
        
        # Price action
        "candle_type": price_action.get("candle_type", ""),
        "candle_size": price_action.get("candle_size", 0),
        "upper_wick": price_action.get("upper_wick", 0),
        "lower_wick": price_action.get("lower_wick", 0),
        
        # Patterns
        "patterns": ", ".join(candle_dict.get("price_patterns", [])),
        
        # LLM analysis
        "market_summary": analysis.get("market_summary", ""),
        "confidence_level": analysis.get("confidence_level", 0),
        "direction": analysis.get("direction", "Neutral"),
        "action": analysis.get("action", "WAIT"),
        "contracts_to_adjust": analysis.get("contracts_to_adjust", 0),
        "reasoning": analysis.get("reasoning", "")
    }
    
    # Create DataFrame with this row
    df_row = pd.DataFrame([data])
    
    # Check if file exists
# Check if file exists
    if os.path.exists(file_path):
        # Read existing Excel file
        try:
            existing_df = pd.read_excel(file_path)
            # Append new row
            updated_df = pd.concat([existing_df, df_row], ignore_index=True)
        except Exception as e:
            print(f"Error reading existing Excel file: {e}")
            updated_df = df_row
    else:
        # Create new DataFrame if file doesn't exist
        updated_df = df_row

    # Save to Excel
    try:
        updated_df.to_excel(file_path, index=False)
        print(f"Logged candle data to {file_path}")
    except Exception as e:
        print(f"Error saving to Excel: {e}")


def update_chat(send_clicks, input_submit, sug1, sug2, sug3, sug4,
               chat_input, chat_messages, chat_history, symbol, timeframe):
    """Updates the chat when the user sends a message"""
    # Check which component triggered the callback
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # If no input was triggered or there's no text input, return current state
    if not triggered_id or (triggered_id == "send-button" and not chat_input):
        return chat_messages, [], {"display": "none"}, chat_history, ""
    
    # Determine the query to be processed
    query = ""
    if triggered_id == "send-button" or triggered_id == "chat-input":
        query = chat_input
    elif triggered_id == "suggestion-1":
        query = "What is the current trend?"
    elif triggered_id == "suggestion-2":
        query = "Show the chart from last week"
    elif triggered_id == "suggestion-3":
        query = "Explain the latest candle pattern"
    elif triggered_id == "suggestion-4":
        query = "What does this support level mean?"
    
    if not query:
        return chat_messages, [], {"display": "none"}, chat_history, ""
    
    # Add user message to the chat
    timestamp = datetime.now().strftime("%H:%M")
    user_message = html.Div([
        html.Div(query, className="user-message"),
        html.Div(timestamp, className="timestamp")
    ])
    
    updated_messages = chat_messages + [user_message]
    
    # Add typing indicator for the assistant
    typing_indicator = html.Div(
        "Analyzing data...",
        className="assistant-message animate-pulse"
    )
    
    updated_messages_with_typing = updated_messages + [typing_indicator]
    
    # Process the query
    response_text = process_chat_query(query, symbol, timeframe)
    
    # Check if the query is about a specific date
    has_specific_date = re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', query)
    has_timeframe_request = re.search(r'\b(daily|weekly|monthly|day|week|month)\b', query.lower())
    show_historical_chart = has_specific_date or has_timeframe_request
    date_to_show = None
    
    if has_specific_date:
        date_match = re.search(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', query)
        if date_match:
            date_to_show = date_match.group(1)
    
    # Format technical terms in the response
    technical_terms = [
        "trend", "support", "resistance", "candle", "pattern", 
        "EMA", "moving average", "ATR", "volatility", "volume", 
        "bullish", "bearish"
    ]
    
    # Create assistant message
    assistant_message = html.Div([
        html.Div(response_text, className="assistant-message"),
        html.Div(timestamp, className="timestamp")
    ])
    
    # Update the messages - FIXED: Check if there's a typing indicator and remove it correctly
    if len(updated_messages) > 0 and isinstance(updated_messages[-1], dict) and updated_messages[-1].get('props', {}).get('className') == "assistant-message animate-pulse":
        updated_messages = updated_messages[:-1]
    
    updated_messages = updated_messages + [assistant_message]
    
    # Prepare visualization if needed
    visual_results = []
    visual_style = {"display": "none"}
    
    if show_historical_chart:
        # Determine the period to get historical data
        if date_to_show:
            # Specific date mentioned
            df = get_historical_data(symbol, timeframe, target_date=date_to_show)
            period_text = f"Historical data for {date_to_show}"
        else:
            # Last week mentioned
            df = get_historical_data(symbol, timeframe, days_ago=7)
            period_text = "Data from last week"
        
        if df is not None and len(df) > 0:
            # Create figure for visualization
            fig = go.Figure()
            
            # Add candlesticks
            fig.add_trace(
                go.Candlestick(
                    x=df['time'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="Price"
                )
            )
            
            # Add EMA9
            fig.add_trace(
                go.Scatter(
                    x=df['time'],
                    y=df['ema9'],
                    name="EMA9",
                    line=dict(color='purple', width=1.5)
                )
            )
            
            # Update layout
            fig.update_layout(
                title=period_text,
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_dark",
                xaxis_rangeslider_visible=False
            )
            
            # Create visual elements
            timeframe_buttons = html.Div([
                html.Div("Change timeframe:", className="me-2"),
                dbc.Button("M5", id="tf-btn-m5", color="outline-info", size="sm", className="timeframe-btn me-1"),
                dbc.Button("M15", id="tf-btn-m15", color="outline-info", size="sm", className="timeframe-btn me-1"),
                dbc.Button("M30", id="tf-btn-m30", color="outline-info", size="sm", className="timeframe-btn me-1"),
                dbc.Button("H1", id="tf-btn-h1", color="outline-info", size="sm", className="timeframe-btn me-1"),
                dbc.Button("D1", id="tf-btn-d1", color="outline-info", size="sm", className="timeframe-btn")
            ], className="timeframe-selector")
            
            # Add close button
            close_button = html.Button(
                "×", 
                id="close-historical-chart",
                style={
                    "position": "absolute",
                    "top": "5px",
                    "right": "5px",
                    "background": "none",
                    "border": "none",
                    "color": "white",
                    "fontSize": "20px",
                    "cursor": "pointer"
                }
            )

            # And when adding it to the visual_results:
            visual_results = [
                html.Div([
                    html.H5(period_text, className="mb-2"),
                    close_button,  # Make sure this is directly included, not wrapped in another html.Div
                    dcc.Graph(figure=fig, config={"displayModeBar": False}),
                    timeframe_buttons
                ], className="historical-chart-container")
            ]
            
            visual_style = {"display": "block", "marginBottom": "20px"}
    
    # Update chat history
    messages_history = chat_history.get("messages", [])
    messages_history.append({
        "role": "user",
        "content": query,
        "timestamp": timestamp
    })
    messages_history.append({
        "role": "assistant",
        "content": response_text,
        "timestamp": timestamp
    })
    
    updated_chat_history = {"messages": messages_history}
    
    # Clear input field
    cleared_input = ""
    
    return updated_messages, visual_results, visual_style, updated_chat_history, cleared_input

# Function to process chat queries
def process_chat_query(query, symbol, timeframe):
    """Processes user queries in the chat using the LLM"""
    global llm_chain
    
    # Check if LLM is initialized
    if llm_chain is None:
        initialize_llm_chain()
    
    # Get historical data for context
    historical_context = get_initial_context(symbol, timeframe, num_candles=20)
    
    # Get current candle data
    current_data = get_current_candle_data(symbol, timeframe)
    
    # Create a prompt specific for chat queries
    chat_prompt = ChatPromptTemplate.from_template("""
    You’re a smart and friendly market analyst, focused on the asset {symbol}.
    Your tone should be natural, a bit informal, but still insightful. Be clear, direct, and avoid long-winded answers.

    # Market Overview
    {context}

    # Current Market Snapshot
    {current_data}

    # User Question
    {query}

    Respond in a conversational and professional way, like you're explaining things to a curious investor or fellow trader.

    • If the user asks about a specific date, mention right away that you're reviewing past data.
    • If they ask for a chart or visual, start with: "[HISTORICAL CHART: date or range]" and then briefly describe what’s visible.
    • If they’re asking about trends, candle behavior, or setups, break it down clearly using what’s in the data.

    Keep it short (max 3 paragraphs), useful, and smooth. No fluff. Just the key points, explained like you're talking to a smart friend.
    """)

    
    # Create a temporary chain for this query
    chat_chain = (
        {"symbol": lambda x: symbol,
         "context": lambda x: historical_context,
         "current_data": lambda x: current_data,
         "query": lambda x: query}
        | chat_prompt
        | ChatOpenAI(
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo"
        )
    )
    
    try:
        # Invoke the chain
        response = chat_chain.invoke({})
        return response.content
        
    except Exception as e:
        print(f"Error processing chat query: {e}")
        import traceback
        traceback.print_exc()
        
        return f"Sorry, I couldn't process your question. Internal error: {str(e)}"

# Function to get historical data for a specific date
def get_historical_data(symbol, timeframe_code, target_date=None, days_ago=None):
    """
    Gets historical data for a specific date or X days ago
    
    Args:
        symbol (str): Asset symbol
        timeframe_code (str): Timeframe code (e.g., "M15")
        target_date (str, optional): Target date in "DD/MM/YYYY" format
        days_ago (int, optional): Number of days to look back
        
    Returns:
        pd.DataFrame: DataFrame with historical data
    """
    if target_date is not None:
        # Convert date string to datetime object
        try:
            # Try various possible date formats
            for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"]:
                try:
                    dt = datetime.strptime(target_date, fmt)
                    break
                except ValueError:
                    continue
            else:
                # If no format works
                raise ValueError(f"Unrecognized date format: {target_date}")
        except Exception as e:
            print(f"Error converting date: {e}")
            return None
    elif days_ago is not None:
        # Calculate target date based on days ago
        dt = datetime.now() - timedelta(days=days_ago)
    else:
        # If neither parameter is provided, use current day
        dt = datetime.now()
    
    # Convert to timestamp
    timestamp = int(dt.timestamp())
    
    # Get historical data (30 candles before and 10 after for context)
    timeframe_mt5 = timeframe_dict.get(timeframe_code, mt5.TIMEFRAME_H4)
    
    # Fetch historical data from a specific date using the timestamp
    rates = mt5.copy_rates_from(symbol, timeframe_mt5, timestamp, 40)
    
    if rates is None or len(rates) == 0:
        print(f"No historical data found for {symbol} on {target_date or f'{days_ago} days ago'}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Calculate indicators
    df['atr'] = calculate_atr(df, period=14)
    df['entropy'] = calculate_directional_entropy(df, period=14)
    df['ema9'] = calculate_ema(df['close'], period=9)
    
    return df

# Callback to handle the close button for the historical chart
@app.callback(
    [Output("chat-visual-results", "children", allow_duplicate=True),
     Output("chat-visual-results", "style", allow_duplicate=True)],
    [Input("close-historical-chart", "n_clicks")],
    [State("chat-visual-results", "children")],
    prevent_initial_call=True
)
def close_historical_chart(n_clicks, current_children):
    """Closes the historical chart when the X button is clicked"""
    if n_clicks:
        return [], {"display": "none"}
    return current_children, {"display": "block"}




def initialize_mt5(server="AMPGlobalUSA-Demo", login=1522209, password="L@X3CgFz"):
    """Initialize connection to MetaTrader5"""
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
    
    # Try to connect without login credentials first (use already connected account)
    if mt5.account_info():
        print("Already connected to MetaTrader5")
        return True
    
    # If not connected, try with credentials
    authorized = mt5.login(login, password, server)
    if not authorized:
        print(f"Failed to connect to account {login}, error code: {mt5.last_error()}")
        return False
    
    print(f"Connected to account {login}")
    return True

def initialize_llm_chain():
    """Initialize LLM chain with modern Langchain approach"""
    global llm_chain
    
        # Initialize LLM
    llm = ChatOpenAI(
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),  # Use the environment variable directly
        model_name="gpt-3.5-turbo"
    )
        
    # Define prompt template moderno


    prompt = ChatPromptTemplate.from_template("""
    You are an expert financial analyst and trader for the futures market. You will analyze market data and help determine trading actions.

    # Market Context
    {context}

    # Current Market Data
    {market_data}

    # Support and Resistance Levels
    {support_resistance}

    # Current Position
    Current contracts: {current_position}
    Maximum allowed contracts: {max_contracts}

    Based on the data provided, analyze the market conditions and determine whether to:
    1. ADD CONTRACTS (if bullish confidence is high)
    2. REMOVE CONTRACTS (if bearish confidence is high)
    3. WAIT (if confidence is low or direction is unclear)

    Your confidence level should be on a scale of -100 to 100, where:
    - Negative values (-100 to -1) indicate bearish confidence (the more negative, the more bearish)
    - Positive values (1 to 100) indicate bullish confidence (the higher, the more bullish)
    - Values close to 0 (-20 to 20) indicate uncertainty

    Provide your reasoning and return your analysis in the following JSON format:
    ```json
    {{
        "market_summary": "Brief description of current market conditions",
        "confidence_level": <number between -100 and 100>,
        "direction": "Bullish/Bearish/Neutral", 
        "action": "ADD_CONTRACTS/REMOVE_CONTRACTS/WAIT",
        "reasoning": "Detailed reasoning for your decision",
        "contracts_to_adjust": <number of contracts to add or remove>
    }}

    """)
    
    # Criar chain moderno
    llm_chain = (
        {"context": RunnablePassthrough(), 
         "market_data": RunnablePassthrough(),
         "support_resistance": RunnablePassthrough(),
         "current_position": RunnablePassthrough(),
         "max_contracts": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    return llm_chain

# Atualize os imports se necessário:
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import json
import re
import numpy as np

# Modify the existing callback to update reasoning and confidence displays
@app.callback(
    [Output("position-display", "children"),
     Output("pnl-display", "children"),
     Output("direction-display", "children"),
     Output("confidence-bar", "value"),
     Output("confidence-bar", "color"),
     Output("reasoning-display", "children"),
     Output("confidence-value", "children"),
     Output("key-factors", "children"),
     Output("last-llm-analysis", "data")],
    [Input("interval-component", "n_intervals"),
     Input("analyze-button", "n_clicks")],
    [State("last-llm-analysis", "data")]
)
def update_dashboard(n_intervals, n_clicks, last_analysis):
    """Update dashboard elements"""
    # Format position display
    position_text = f"{current_position} Contracts"
    
    # Format P&L display
    pnl_text = f"${total_pnl:.2f}"
    
    # Format direction display
    direction_text = market_direction
    
    # Calculate confidence bar value (convert from -100/100 scale to 0/100 scale)
    confidence_bar_value = (confidence_level + 100) / 2
    
    # Determine bar color based on confidence level
    # if confidence_level < -50:
    #     bar_color = "danger"  # Red for very bearish
    # elif confidence_level < -20:
    #     bar_color = "warning"  # Yellow for slightly bearish
    # elif confidence_level < 20:
    #     bar_color = "info"     # Blue for neutral
    # elif confidence_level < 50:
    #     bar_color = "primary"  # Light green for slightly bullish
    # else:
    #     bar_color = "success"  # Green for very bullish
    # Determine bar color based on confidence level
    if confidence_level < 0:
        bar_color = "danger"  # Vermelho para qualquer valor negativo (bearish)
    elif confidence_level > 0:
        bar_color = "success"  # Verde para qualquer valor positivo (bullish)
    else:
        bar_color = "info"     # Azul para neutro (exatamente zero)
    # Format reasoning text with improved styling
    reasoning_html = []
    
    if llm_reasoning:
        # Split reasoning into paragraphs
        paragraphs = llm_reasoning.split("\n")
        for para in paragraphs:
            if para.strip():  # Skip empty paragraphs
                reasoning_html.append(html.P(para))
    else:
        reasoning_html = html.P("No analysis available yet. Start trading or trigger analysis manually.")
    
    # Format confidence value text
    if confidence_level > 0:
        confidence_text = f"Bullish Confidence: +{confidence_level}"
        confidence_style = {"color": "green"}
    elif confidence_level < 0:
        confidence_text = f"Bearish Confidence: {confidence_level}"
        confidence_style = {"color": "red"}
    else:
        confidence_text = f"Neutral Confidence: {confidence_level}"
        confidence_style = {"color": "blue"}
    
    confidence_value_html = html.Span(confidence_text, style=confidence_style)
    
    # Extract key factors from the reasoning
    key_factors_html = extract_key_factors(llm_reasoning)
    
    # Store the latest analysis for reference
    current_analysis = {
        "reasoning": llm_reasoning,
        "confidence_level": confidence_level,
        "direction": market_direction,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return position_text, pnl_text, direction_text, confidence_bar_value, bar_color, reasoning_html, confidence_value_html, key_factors_html, current_analysis

def extract_key_factors(reasoning_text):
    """Extract key factors from reasoning text and format as bullet points"""
    if not reasoning_text:
        return html.P("No factors available", className="text-muted")
    
    # Tente identificar padrões comuns que indiquem fatores-chave
    factors = []
    
    # Padrões a procurar
    patterns = [
        r"price.+support",
        r"price.+resistance",
        r"trend",
        r"volatility",
        r"volume",
        r"pattern",
        r"indicator",
        r"market condition",
        r"ATR",
        r"entropy",
        r"EMA",
        r"momentum",
        r"bullish signal",
        r"bearish signal"
    ]
    
    # Divida o texto de raciocínio em frases
    sentences = re.split(r'[.!?]+', reasoning_text)
    
    # Examine cada frase para ver se contém um dos padrões
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        for pattern in patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                # Evite duplicatas
                if sentence not in factors:
                    factors.append(sentence)
                break
    
    # Se não encontrou fatores específicos, use algumas frases do raciocínio
    if not factors and sentences:
        # Use até 3 sentenças significativas do raciocínio
        meaningful_sentences = [s for s in sentences if len(s.split()) > 5][:3]
        factors = meaningful_sentences
    
    # Formatar como lista de bullet points
    if factors:
        return html.Ul([html.Li(factor) for factor in factors], className="mt-2")
    else:
        return html.P("Analysis doesn't contain specific factors", className="text-muted")

# Modificação na função analyze_market para extrair mais informações
def analyze_market(symbol, timeframe):
    """Analyze market using LLM"""
    global llm_reasoning, confidence_level, market_direction, llm_chain
    global initial_market_context, use_initial_context_enabled
    
    # Verify positions and get position info
    positions = mt5.positions_get(symbol=symbol)
    position_info = ""
    
    if positions and len(positions) > 0:
        for pos in positions:
            position_type = "Long" if pos.type == mt5.ORDER_TYPE_BUY else "Short"
            position_info += f"MT5 Position: {position_type} {pos.volume} contracts at {pos.price_open}, P&L: {pos.profit}\n"
    
    # Get support resistance levels as string
    sr_str = "\n".join([f"Level {i+1}: {level}" for i, level in enumerate(support_resistance_levels)])
    
    # Get current market data
    market_data = get_current_candle_data(symbol, timeframe)
    
    # Add position information to context if available
    if position_info:
        # Try to add to context as JSON data
        try:
            context_dict = json.loads(market_data)
            context_dict["mt5_positions"] = position_info
            market_data = json.dumps(context_dict)
        except:
            # If that fails, just keep context as is
            pass
    
    # Prepare context data, including historical context if enabled
    if use_initial_context_enabled and initial_market_context:
        # Get a small amount of recent context to supplement the historical view
        recent_data = get_initial_context(symbol, timeframe, num_candles=2) #________________________________________________________________________________quantia de candles da analize a cada novo candle
        
        # Combine historical and recent context
        context = f"""
# Long-Term Historical Market Analysis (H4 Timeframe)
{initial_market_context}

# Recent Market Context ({timeframe} Timeframe)
{recent_data}
"""
    else:
        # If not using initial context, just use standard context
        context = get_initial_context(symbol, timeframe, num_candles=2)
    
    # Run LLM analysis using the existing chain
    try:
        # Use the existing llm_chain
        response = llm_chain.invoke({
            "context": context,
            "market_data": market_data,
            "support_resistance": sr_str,
            "current_position": current_position,
            "max_contracts": max_contracts
        })
        
        # Process response as before
        response_text = response.content
        analysis = parse_llm_response(response_text)
        
        # Update global variables
        llm_reasoning = analysis.get('reasoning', "No reasoning provided")
        confidence_level = analysis.get('confidence_level', 0)
        market_direction = analysis.get('direction', "Neutral")
        print(f"Full LLM reasoning: {llm_reasoning}")

        # Return analysis
        return analysis
    except Exception as e:
        print(f"Error in analyze_market: {e}")
        import traceback
        traceback.print_exc()
        return {
            "market_summary": "Error analyzing market",
            "confidence_level": 0,
            "direction": "Neutral",
            "action": "WAIT",
            "reasoning": f"Error in LLM analysis: {str(e)}",
            "contracts_to_adjust": 0
        }
def update_trade_history(n_intervals):
    """Update trade history display"""
    if not trade_history:
        return html.P("No trades executed yet", className="text-muted")
    
    # Create trade history cards
    history_cards = []
    
    for i, trade in enumerate(reversed(trade_history)):  # Mostrar mais recentes primeiro
        # Determine card color based on action
        card_color = "success" if trade["action"] == "ADD_CONTRACTS" else "danger"
        
        # Format timestamp
        timestamp = trade["timestamp"]
        
        # Format PnL with color
        pnl_color = "text-success" if trade["pnl_change"] >= 0 else "text-danger"
        pnl_text = f"+${trade['pnl_change']:.2f}" if trade["pnl_change"] >= 0 else f"-${abs(trade['pnl_change']):.2f}"
        
        # Create card
        card = dbc.Card([
            dbc.CardHeader(f"Trade #{len(trade_history) - i} - {timestamp}"),
            dbc.CardBody([
                html.H5(f"{trade['action']}", className=f"text-{card_color}"),
                html.P([
                    f"Contracts: {trade['contracts']} @ {trade['price']:.5f}",
                    html.Br(),
                    html.Span(f"P&L: {pnl_text}", className=pnl_color)
                ])
            ])
        ], className="mb-2")
        
        history_cards.append(card)
    
    return history_cards
def get_initial_context(symbol, timeframe, num_candles=3):
    """Get initial market context for LLM"""
    if timeframe not in timeframe_dict:
        return "Invalid timeframe"
    
    # Get historical data
    rates = mt5.copy_rates_from_pos(symbol, timeframe_dict[timeframe], 0, num_candles)
    
    if rates is None or len(rates) == 0:
        return "No data available"
    
    # Convert to dataframe
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Calculate indicators
    df['atr'] = calculate_atr(df, period=14)
    df['entropy'] = calculate_directional_entropy(df, period=14)
    df['ema9'] = calculate_ema(df['close'], period=9)
    
    # Detect price patterns
    patterns = detect_price_patterns(df)
    
    # Prepare context summary
    market_summary = {
        "period_analyzed": f"{num_candles} {timeframe} candles",
        "start_date": df['time'].iloc[0].strftime("%Y-%m-%d %H:%M"),
        "end_date": df['time'].iloc[-1].strftime("%Y-%m-%d %H:%M"),
        "price_range": {
            "high": float(df['high'].max()),
            "low": float(df['low'].min()),
            "current": float(df['close'].iloc[-1])
        },
        "trend_summary": {
            "direction": "Bullish" if df['close'].iloc[-1] > df['open'].iloc[0] else "Bearish",
            "strength": abs(float(df['close'].iloc[-1] - df['open'].iloc[0])) / float(df['atr'].iloc[-1]) if not np.isnan(df['atr'].iloc[-1]) and df['atr'].iloc[-1] != 0 else 0,
        },
        "volatility": {
            "average_atr": float(df['atr'].mean()) if not np.isnan(df['atr'].mean()) else 0,
            "entropy": float(df['entropy'].mean()) if not np.isnan(df['entropy'].mean()) else 0
        },
        "key_levels": {
            "recent_high": float(df['high'].max()),
            "recent_low": float(df['low'].min()),
            "ema9": float(df['ema9'].iloc[-1]) if not np.isnan(df['ema9'].iloc[-1]) else 0
        },
        "price_patterns": patterns
    }
    
    return json.dumps(market_summary, indent=2)

def get_current_candle_data(symbol, timeframe):
    """Get current candle data"""
    if timeframe not in timeframe_dict:
        return "Invalid timeframe"
    
    # Get current candle
    rates = mt5.copy_rates_from_pos(symbol, timeframe_dict[timeframe], 0, 2)
    
    if rates is None or len(rates) < 2:
        return "No data available"
    
    # Convert to dataframe
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Calculate indicators
    atr = calculate_atr(df, period=14)
    entropy = calculate_directional_entropy(df, period=14)
    ema9 = calculate_ema(df['close'], period=9)
    
    # Detect price patterns
    patterns = detect_price_patterns(df)
    
    # Prepare data summary
    candle_data = {
        "timestamp": df['time'].iloc[-1].strftime("%Y-%m-%d %H:%M"),
        "ohlc": {
            "open": float(df['open'].iloc[-1]),
            "high": float(df['high'].iloc[-1]),
            "low": float(df['low'].iloc[-1]),
            "close": float(df['close'].iloc[-1])
        },
        "volume": float(df['tick_volume'].iloc[-1]),
        "indicators": {
            "atr": float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0,
            "directional_entropy": float(entropy.iloc[-1]) if not np.isnan(entropy.iloc[-1]) else 0,
            "ema9": float(ema9.iloc[-1]) if not np.isnan(ema9.iloc[-1]) else 0
        },
        "price_action": {
            "candle_type": "Bullish" if df['close'].iloc[-1] > df['open'].iloc[-1] else "Bearish",
            "candle_size": abs(float(df['close'].iloc[-1] - df['open'].iloc[-1])),
            "upper_wick": float(df['high'].iloc[-1]) - max(float(df['open'].iloc[-1]), float(df['close'].iloc[-1])),
            "lower_wick": min(float(df['open'].iloc[-1]), float(df['close'].iloc[-1])) - float(df['low'].iloc[-1])
        },
        "price_patterns": patterns
    }
    
    return json.dumps(candle_data, indent=2)

def execute_trade(symbol, action, contracts_to_adjust):
    """Execute trade on MetaTrader5"""
    global current_position, total_pnl, trade_history
    
    # Check if we need to adjust position
    if action == "WAIT" or contracts_to_adjust == 0:
        return "No trade executed"
    
    # Get current symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return f"Symbol {symbol} not found"
    
    # Verificar posições existentes primeiro
    positions = mt5.positions_get(symbol=symbol)
    print(f"Current positions for {symbol}: {len(positions) if positions else 0}")
    
    # Log position details if they exist
    if positions and len(positions) > 0:
        for position in positions:
            print(f"Position: {position.identifier}, Type: {'Buy' if position.type == mt5.ORDER_TYPE_BUY else 'Sell'}, Volume: {position.volume}, Price: {position.price_open}")
    
    # Prepare trade request
    trade_type = mt5.ORDER_TYPE_BUY if action == "ADD_CONTRACTS" else mt5.ORDER_TYPE_SELL
    price = symbol_info.ask if trade_type == mt5.ORDER_TYPE_BUY else symbol_info.bid
    
    # Determine volume based on contracts to adjust
    volume = abs(contracts_to_adjust)
    
    # Create trade request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": trade_type,
        "price": price,
        "deviation": 20,  # Price deviation in points
        "magic": 123456,  # Magic number for identifying trades
        "comment": f"LLM Bot {'Buy' if trade_type == mt5.ORDER_TYPE_BUY else 'Sell'}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Send trade order
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        error_message = f"Order failed, retcode={result.retcode}"
        print(error_message)
        return error_message
    else:
        success_message = f"Order executed successfully: {action}, {contracts_to_adjust} contracts at {price}"
        print(success_message)
    
    # Update position tracking
    if action == "ADD_CONTRACTS":
        current_position += contracts_to_adjust
    else:  # REMOVE_CONTRACTS
        current_position -= contracts_to_adjust
    
    # Get actual PnL from MetaTrader if available
    pnl_change = 0
    positions = mt5.positions_get(symbol=symbol)
    if positions and len(positions) > 0:
        # Calcular P&L real com base nas posições existentes
        pnl_change = sum(pos.profit for pos in positions)
        total_pnl = pnl_change
        print(f"Actual P&L from positions: {pnl_change}")
    else:
        # Se não há posições, usar cálculo simulado
        pnl_change = np.random.normal(0, 10) * contracts_to_adjust  # Simulated PnL change
        total_pnl += pnl_change
        print(f"Simulated P&L change: {pnl_change}")
    
    # Add to trade history
    new_trade = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "contracts": contracts_to_adjust,
        "price": price,
        "pnl_change": pnl_change
    }
    trade_history.append(new_trade)
    print(f"Trade added to history: {new_trade}")
    
    return success_message
def trading_loop(symbol, timeframe):
    """Main trading loop"""
    global running, trade_history, current_position, total_pnl
    
    print(f"Starting trading loop for {symbol} on {timeframe} timeframe")
    
    # Keep track of the last processed candle time
    last_candle_time = None
    
    # Verify existing positions at startup
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        print(f"Found {len(positions)} existing positions for {symbol}")
        current_position = sum(pos.volume for pos in positions)
        print(f"Setting current position to {current_position} contracts")
    
    while running:
        try:
            # Check if a new candle has formed
            current_rates = mt5.copy_rates_from_pos(symbol, timeframe_dict.get(timeframe, mt5.TIMEFRAME_H4), 0, 1)
            if current_rates is not None and len(current_rates) > 0:
                current_candle_time = pd.to_datetime(current_rates[0]['time'], unit='s')
                
                # If this is a new candle, analyze the market
                if last_candle_time is None or current_candle_time > last_candle_time:
                    print(f"New candle detected at {current_candle_time}")
                    last_candle_time = current_candle_time
                    
                    # Get current candle data
                    candle_data = get_current_candle_data(symbol, timeframe)
                    
                    # Perform market analysis
                    analysis = analyze_market(symbol, timeframe)
                    print(f"Market analysis: {analysis['market_summary']}")
                    print(f"Confidence: {analysis['confidence_level']}, Direction: {analysis['direction']}")
                    
                    # Log candle data and analysis to Excel
                    log_candle_data(symbol, timeframe, candle_data, analysis)
                    # If confidence exceeds threshold, execute automatically
                    if abs(analysis['confidence_level']) > 50:  # Confidence threshold
                        print(f"Confidence level {analysis['confidence_level']} exceeds threshold. Executing trade...")
                        contracts_to_adjust = min(
                            analysis['contracts_to_adjust'],
                            max_contracts - current_position if analysis['action'] == "ADD_CONTRACTS" else current_position
                        )
                        
                        if contracts_to_adjust > 0:
                            result = execute_trade(symbol, analysis['action'], contracts_to_adjust)
                            print(result)
                    else:
                        print(f"Confidence level {analysis['confidence_level']} below threshold. Waiting for user input.")
            
            # Check if there are user inputs in the queue
            try:
                user_input = trade_queue.get_nowait()
                print(f"Processing user input: {user_input}")
                
                # Get current candle data
                candle_data = get_current_candle_data(symbol, timeframe)
                
                # Analyze market
                analysis = analyze_market(symbol, timeframe)
                
                # Log candle data and analysis to Excel (for manual analysis as well)
                log_candle_data(symbol, timeframe, candle_data, analysis, file_path=f"metatradebot_manual_log_{symbol}_{timeframe}.xlsx")
                
                # Execute trade based on analysis
                if analysis['action'] != "WAIT":
                    # Limit contract adjustments to respect max_contracts
                    contracts_to_adjust = min(
                        analysis['contracts_to_adjust'],
                        max_contracts - current_position if analysis['action'] == "ADD_CONTRACTS" else current_position
                    )
                    
                    if contracts_to_adjust > 0:
                        result = execute_trade(symbol, analysis['action'], contracts_to_adjust)
                        print(result)
                        
                        # Calculate position performance
                        symbol_info = mt5.symbol_info(symbol)
                        if symbol_info:
                            current_price = symbol_info.ask
                            performance = calculate_position_performance(trade_history, current_price)
                            total_pnl = performance['total_pnl']
                            
                            print(f"Position size: {performance['position_size']}")
                            print(f"Average entry: {performance['average_entry']:.5f}")
                            print(f"Unrealized P&L: ${performance['unrealized_pnl']:.2f}")
                            print(f"Total P&L: ${performance['total_pnl']:.2f}")
                    else:
                        print(f"Skipping trade - contracts to adjust ({contracts_to_adjust}) <= 0")
                else:
                    print("Analysis recommends waiting - no trade executed")
            
            except queue.Empty:
                # No user input, continue with regular updates
                pass
            
            # Update positions every 30 seconds
            if int(time.time()) % 30 == 0:
                # Check positions in MetaTrader5
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    mt_position_size = sum(pos.volume for pos in positions)
                    if mt_position_size != current_position:
                        print(f"Position discrepancy detected: MT5={mt_position_size}, Bot={current_position}")
                        print(f"Updating internal position to match MT5: {mt_position_size}")
                        current_position = mt_position_size
                        
                    # Update P&L with real data
                    mt_pnl = sum(pos.profit for pos in positions)
                    if abs(mt_pnl - total_pnl) > 0.01:  # If there's a significant difference
                        print(f"P&L discrepancy detected: MT5=${mt_pnl:.2f}, Bot=${total_pnl:.2f}")
                        print(f"Updating internal P&L to match MT5: ${mt_pnl:.2f}")
                        total_pnl = mt_pnl
                else:
                    # If there are no positions, and the bot thinks it still has, adjust
                    if current_position != 0:
                        print(f"No positions found in MT5, but bot thinks it has {current_position}. Resetting to 0.")
                        current_position = 0
                        total_pnl = 0
            
            # Sleep to avoid excessive polling
            time.sleep(1)
            
        except Exception as e:
            print(f"Error in trading loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5) 
            
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("MetaTrader5 LLM Trading Bot", className="text-center my-4"),
        ])
    ]),
    # Somewhere in your main layout
    html.Button(
        id="close-historical-chart", 
        style={"display": "none"}  # Hidden initially
    ),
    dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader([
                html.H4("Trading Assistant", className="d-inline me-2"),
                html.Span("Ask about the asset, technical analysis, or specific periods", 
                         className="text-muted small")
            ]),
            dbc.CardBody([
                # Chat display area
                html.Div(
                    id="chat-messages",
                    className="chat-container mb-3",
                    style={
                        "height": "250px",
                        "overflowY": "auto",
                        "display": "flex",
                        "flexDirection": "column",
                        "padding": "10px",
                        "border": "1px solid rgba(255, 255, 255, 0.1)",
                        "borderRadius": "8px",
                        "backgroundColor": "rgba(0, 0, 0, 0.2)"
                    },
                    children=[
                        html.Div(
                            "Hello! I'm your trading assistant. Ask me about the current asset, technical analysis, or about specific periods like",
                            className="assistant-message"
                        )
                    ]
                ),
                
                # Area for visual results
                html.Div(
                    id="chat-visual-results",
                    className="mb-3",
                    style={"display": "none"}
                ),
                
                # User input area
                dbc.InputGroup([
                    dbc.Input(
                        id="chat-input",
                        placeholder="Type your question about the asset...",
                        type="text",
                        className="border-primary",
                        n_submit=0
                    ),
                    dbc.Button(
                        html.Div([
                            "Send ",
                            html.I(className="fas fa-arrow-right", style={"marginLeft": "5px"})
                        ], style={"display": "flex", "alignItems": "center"}) ,
                        id="send-button",
                        color="primary",
                        className="px-3",
                        style={"fontWeight": "bold", "borderTopLeftRadius": "0", "borderBottomLeftRadius": "0"},
                        n_clicks=0,
                    )
                ]),
                
                # Suggestion buttons
                html.Div([
                    html.P("Suggestions:", className="mt-2 mb-1 text-muted small"),
                    html.Div([
                        dbc.Button("Could you provide me with an analysis of how the market was 5 hours ago?", 
                                 id="suggestion-1", 
                                 color="light", 
                                 size="sm", 
                                 className="me-2 mb-2 suggestion-btn"),
                        dbc.Button("Show the chart from last week", 
                                 id="suggestion-2", 
                                 color="light", 
                                 size="sm", 
                                 className="me-2 mb-2 suggestion-btn"),
                        dbc.Button("What is the current trend?", 
                                 id="suggestion-3", 
                                 color="light", 
                                 size="sm", 
                                 className="me-2 mb-2 suggestion-btn"),
                        dbc.Button("What does this support level mean?", 
                                 id="suggestion-4", 
                                 color="light", 
                                 size="sm", 
                                 className="me-2 mb-2 suggestion-btn"),
                    ], className="d-flex flex-wrap")
                ])
            ])
        ], className="mb-4 chat-card shadow")
    ], width=12)
]),

    # Add this Div to store chat history
    dcc.Store(id="chat-history", data={"messages": []}),

    # Adicione este Div para armazenar o histórico de conversas
    dcc.Store(id="chat-history", data={"messages": []}),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trading Setup"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Asset Symbol:"),
                            dbc.Input(id="symbol-input", type="text", value="NGEM25", placeholder="Enter symbol (e.g., EURUSD)"),
                        ], width=6),
                        dbc.Col([
                            html.Label("Timeframe:"),
                            dcc.Dropdown(
                                id="timeframe-dropdown",
                                options=[
                                    {"label": "1 Minute (M1)", "value": "M1"},
                                    {"label": "5 Minutes (M5)", "value": "M5"},
                                    {"label": "15 Minutes (M15)", "value": "M15"},
                                    {"label": "30 Minutes (M30)", "value": "M30"},
                                    {"label": "1 Hour (H1)", "value": "H1"},
                                    {"label": "4 Hours (H4)", "value": "H4"},
                                    {"label": "1 Day (D1)", "value": "D1"},
                                ],
                                value="H4"
                            ),
                        ], width=6),
                    ]),
                    
                    html.Br(),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Support & Resistance Levels:"),
                            dbc.Input(id="sr-input", type="text", placeholder="Enter levels separated by commas (e.g., 1.1000, 1.1050)"),
                            html.Div(id="sr-output", className="mt-2")
                        ], width=6),
                        dbc.Col([
                            html.Label("Max Contracts:"),
                            dbc.Input(id="max-contracts-input", type="number", value=5, min=1, max=100),
                        ], width=6),
                    ]),
                    
                    html.Br(),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Checkbox(id="use-context-checkbox", label="Use Initial Market Context", value=False),
                            dbc.Tooltip(
                                "Enabling this will analyze previous candles to provide context for the LLM, but will use more tokens.",
                                target="use-context-checkbox",
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Button("Start", id="start-button", color="success", className="me-2"),
                            dbc.Button("Stop", id="stop-button", color="danger", className="me-2"),
                            dbc.Button("Trigger Analysis", id="analyze-button", color="primary", className="me-2"),
                        ], width=6, className="d-flex justify-content-end align-items-end"),
                    ]),
                ]),
            ], className="mb-4"),
        ], width=12),
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Market Dashboard"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Current Position:"),
                            html.H3(id="position-display", children="0 Contracts"),
                        ], width=4),
                        dbc.Col([
                            html.H5("Total P&L:"),
                            html.H3(id="pnl-display", children="$0.00"),
                        ], width=4),
                        dbc.Col([
                            html.H5("Market Direction:"),
                            html.H3(id="direction-display", children="Neutral"),
                        ], width=4),
                    ]),
                    
                    html.Hr(),
                    
                    # Add price chart
                    html.Div([
                        html.H5("Price Chart:"),
                        dcc.Graph(id="price-chart", style={"height": "400px"}),
                    ], className="mb-4"),
                    
                    # Nova seção para exibir o raciocínio do LLM e nível de confiança
                    dbc.Card([
                        dbc.CardHeader("LLM Analysis for Current Candle"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Confidence Level:", className="mb-2"),
                                    # Barra de confiança com gradiente de cores
                                    html.Div([
                                        dbc.Progress(
                                            id="confidence-bar",
                                            value=50,
                                            color="info",
                                            className="mb-1",
                                            style={"height": "30px"}
                                        ),
                                        html.Div(
                                            id="confidence-scale",
                                            className="d-flex justify-content-between mt-1",
                                            children=[
                                                html.Span("Bearish Confidence", className="text-danger"),
                                                html.Span("Neutral", className="text-info"),
                                                html.Span("Bullish Confidence", className="text-success")
                                            ]
                                        )
                                    ], className="mb-3"),
                                    html.Div(
                                        id="confidence-value",
                                        className="text-center mb-3 h7"
                                    ),
                                ], width=12),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.H5("LLM Reasoning:"),
                                    dbc.Card(
                                        dbc.CardBody(
                                            html.Div(
                                                id="reasoning-display",
                                                className="border-0 text-white"
                                            )
                                        ),
                                        className="bg-dark text-light mt-2 mb-3" 
                                    ),
                                ], width=12),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Key Factors Influencing Decision:"),
                                    html.Div(
                                        id="key-factors",
                                        className="mt-2"
                                    ),
                                ], width=12),
                            ]),
                        ]),
                    ], className="mb-4"),
                    
                ]),
            ], className="mb-4"),
        ], width=12),
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trade History"),
                dbc.CardBody([
                    html.Div(id="trade-history-display", style={"maxHeight": "300px", "overflow": "auto"})
                ]),
            ]),
        ], width=12),
    ]),
    
    # Add interval component for regular updates
    dcc.Interval(
        id='interval-component',
        interval=2*1000,  # in milliseconds (5 seconds)
        n_intervals=0
    ),
    
    # Store de dados para o último análise do LLM
    dcc.Store(id="last-llm-analysis", data={}),
])
@app.callback(
    [Output("sr-output", "children")],
    [Input("sr-input", "value")]
)
def update_sr_levels(sr_input):
    """Update support and resistance levels"""
    global support_resistance_levels
    
    if not sr_input:
        support_resistance_levels = []
        return [html.P("No levels defined", className="text-muted")]
    
    try:
        # Parse comma-separated values
        levels = [float(level.strip()) for level in sr_input.split(",") if level.strip()]
        support_resistance_levels = sorted(levels)
        
        # Create output display
        level_displays = []
        for i, level in enumerate(support_resistance_levels):
            level_displays.append(html.Span(f"{level:.4f}", className="badge bg-primary me-2"))
        
        return [html.Div(level_displays)]
    
    except ValueError:
        return [html.P("Invalid input. Use comma-separated numbers.", className="text-danger")]

@app.callback(
    Output("price-chart", "figure"),
    [Input("interval-component", "n_intervals")],
    [State("symbol-input", "value"),
     State("timeframe-dropdown", "value")]
)
def update_price_chart(n_intervals, symbol, timeframe):
    """Update price chart with current market data"""
    if not symbol or not timeframe:
        # Return empty chart if no symbol or timeframe
        return go.Figure()
    
    # Define the timeframe mapping
    timeframe_dict = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    
    try:
        # Check if mt5 is initialized
        if not mt5.terminal_info():
            # Return placeholder chart if MT5 not initialized
            fig = go.Figure()
            fig.add_annotation(
                text="MetaTrader5 not initialized. Start trading to view chart.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            return fig
        
        # Get 50 most recent candles for the chart
        rates = mt5.copy_rates_from_pos(symbol, timeframe_dict.get(timeframe, mt5.TIMEFRAME_H4), 0, 50)
        
        if rates is None or len(rates) == 0:
            # Return empty chart if no data
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for {symbol} on {timeframe} timeframe",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            return fig
        
        # Convert to dataframe
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Calculate indicators
        df['atr'] = calculate_atr(df, period=14)
        df['entropy'] = calculate_directional_entropy(df, period=14)
        df['ema9'] = calculate_ema(df['close'], period=9)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price", "Indicators")
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['time'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add EMA
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['ema9'],
                name="EMA9",
                line=dict(color='purple', width=1)
            ),
            row=1, col=1
        )
        
        # Add support and resistance lines
        for level in support_resistance_levels:
            fig.add_shape(
                type="line",
                x0=df['time'].iloc[0],
                x1=df['time'].iloc[-1],
                y0=level,
                y1=level,
                line=dict(color="rgba(255, 0, 0, 0.5)", width=2, dash="dash"),
                row=1, col=1
            )
        
        # Add ATR
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['atr'],
                name="ATR (14)",
                line=dict(color='orange', width=1)
            ),
            row=2, col=1
        )
        
        # Add Directional Entropy
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['entropy'],
                name="Directional Entropy",
                line=dict(color='blue', width=1)
            ),
            row=2, col=1
        )
        
        # Add volume as bar chart
        fig.add_trace(
            go.Bar(
                x=df['time'],
                y=df['tick_volume'],
                name="Volume",
                marker=dict(
                    color=np.where(df['close'] > df['open'], 'green', 'red'),
                    opacity=0.5
                )
            ),
            row=2, col=1
        )
        
        # Add trade entry points if available
        for trade in trade_history:
            timestamp = datetime.strptime(trade['timestamp'], "%Y-%m-%d %H:%M:%S")
            
            # Find the closest candle to this timestamp
            closest_idx = np.abs(df['time'] - timestamp).argmin()
            
            if closest_idx >= 0 and closest_idx < len(df):
                # Add marker for trade entry
                marker_color = 'green' if trade['action'] == 'ADD_CONTRACTS' else 'red'
                
                fig.add_trace(
                    go.Scatter(
                        x=[df['time'].iloc[closest_idx]],
                        y=[df['high'].iloc[closest_idx] * 1.002 if trade['action'] == 'ADD_CONTRACTS' else df['low'].iloc[closest_idx] * 0.998],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down' if trade['action'] == 'REMOVE_CONTRACTS' else 'triangle-up',
                            size=12,
                            color=marker_color,
                            line=dict(width=2, color='black')
                        ),
                        name=f"{trade['action']} ({trade['contracts']})"
                    ),
                    row=1, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - {timeframe}",
            xaxis_title="Time",
            height=600,
            margin=dict(l=50, r=50, t=80, b=50),
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        # Set y-axis titles
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Indicators/Volume", row=2, col=1)
        
        return fig
    
    except Exception as e:
        print(f"Error updating chart: {e}")
        
        # Return error message in chart
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error updating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

@app.callback(
    [Output("max-contracts-input", "disabled")],
    [Input("start-button", "n_clicks")],
    [State("max-contracts-input", "value")]
)
def update_max_contracts(n_clicks, max_contracts_value):
    """Update max contracts setting"""
    global max_contracts
    
    if n_clicks is not None and n_clicks > 0:
        max_contracts = max(1, min(100, max_contracts_value))
        return [True]  # Disable input after starting
    
    return [False]

@app.callback(
    [Output("start-button", "disabled"),
     Output("stop-button", "disabled"),
     Output("analyze-button", "disabled")],
    [Input("start-button", "n_clicks"),
     Input("stop-button", "n_clicks")],
    [State("symbol-input", "value"),
     State("timeframe-dropdown", "value"),
     State("use-context-checkbox", "value")]
)
def control_trading(start_clicks, stop_clicks, symbol, timeframe, use_context):
    """Control trading process"""
    global running, llm_chain, trade_queue, initial_market_context, use_initial_context_enabled
    
    # Get the button that triggered the callback
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    if triggered_id == "start-button" and start_clicks is not None and start_clicks > 0:
        # Initialize components
        if not initialize_mt5():
            return [False, True, False]
        
        # Initialize LLM chain
        if llm_chain is None:
            initialize_llm_chain()
        
        # Set the global flag for using initial context
        use_initial_context_enabled = bool(use_context)
        
        # If enabled, get and analyze initial context ONCE at startup
        if use_initial_context_enabled:
            try:
                print("Getting initial market context...")
                # Get historical data for H4 timeframe
                historical_data = get_initial_context(symbol, "H4", num_candles=15) #____________________________________________________________ quantia de candles para a analize H4 do langchain
                
                # Use your existing initial_context_prompt from prompts.py
                from prompts import initial_context_prompt
                
                # Create LLM for historical analysis
                history_llm = ChatOpenAI(
                    temperature=0.1,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="gpt-3.5-turbo"
                )
                
                # Create a chain for historical analysis using your existing prompt
                historical_chain = (
                    {"historical_data": lambda x: historical_data}
                    | initial_context_prompt 
                    | history_llm
                )
                
                # Run the historical analysis
                response = historical_chain.invoke({})
                
                # Store the result in the global variable
                if isinstance(response.content, str):
                    initial_market_context = response.content
                else:
                    # Format the JSON response for display in the prompt
                    initial_market_context = json.dumps(response.content, indent=2)
                
                print("Initial market context analysis completed and stored.")
                
            except Exception as e:
                print(f"Error getting initial context: {e}")
                import traceback
                traceback.print_exc()
                initial_market_context = None
        else:
            # Reset the initial context if not using it
            initial_market_context = None
        
        # Start trading loop
        running = True
        trading_thread = threading.Thread(target=trading_loop, args=(symbol, timeframe))
        trading_thread.daemon = True
        trading_thread.start()
        
        return [True, False, False]
    
    elif triggered_id == "stop-button" and stop_clicks is not None and stop_clicks > 0:
        # Stop trading loop
        running = False
        
        # Shutdown MT5
        mt5.shutdown()
        
        return [False, True, True]
    
    # Default state
    return [False, True, True]
@app.callback(
    Output("analyze-button", "n_clicks"),
    [Input("analyze-button", "n_clicks")],
    [State("symbol-input", "value"),
     State("timeframe-dropdown", "value")]
)
def trigger_analysis(n_clicks, symbol, timeframe):
    """Trigger manual market analysis"""
    if n_clicks is not None and n_clicks > 0:
        # Add to the queue
        trade_queue.put("ANALYZE")
    
    return 0  # Reset clicks
# Inside analyze_market function after getting the response

if __name__ == "__main__":
    app.run(debug=True, port=8050)
