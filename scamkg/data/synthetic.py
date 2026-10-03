"""Synthetic, template-generated phone calls for the offline demo and the test-suite.

These are NOT real data and results on them are not evidence for the research questions. They
exist so the whole pipeline (extraction -> KG -> features -> models -> evaluation -> dashboard) can
run end-to-end before the real data sets, an LLM server or Neo4j are available.

Three synthetic "data sets" mimic the proposal's setup:
* demo_primary  - balanced dialogues, style A ("caller: ... receiver: ...")      [train/val/test]
* demo_shifted  - balanced dialogues, style B (different wording, "Speaker 1:") [cross-dataset]
* demo_robocall - scam robocall monologues with IVR prompts ("press 1")          [real-world-like]
Every call carries its template's gold triples (used as X1 gold for the demo).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scamkg.utils import get_logger, write_jsonl

log = get_logger(__name__)


def CL(v, d=""): return ("CLAIMS_TO_BE", "ClaimedIdentity", v, d)
def ORG(v): return ("MENTIONS_ORG", "Organization", v, "")
def TAC(v, d=""): return ("USES_TACTIC", "PressureTactic", v, d)
def REQ(v, d=""): return ("REQUESTS", "RequestedAsset", v, d)
def PAY(v, d=""): return ("PAY_VIA", "PaymentChannel", v, d)
def NUM(v="{phone}"): return ("PROVIDES_NUMBER", "CallbackNumber", v, "")


@dataclass
class Step:
    role: str                        # "C" caller / "R" receiver
    a: list                          # phrases for style A (str or (str, [triples]))
    b: list | None = None            # phrases for style B (falls back to a)
    triples: list = field(default_factory=list)
    p: float = 1.0


@dataclass
class Script:
    name: str
    label: int
    label3: str
    steps: list


SLOTS = {
    "agent": ["Brown", "Johnson", "Miller", "Davis", "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Martin"],
    "first": ["Sarah", "Mike", "Emily", "David", "Jessica", "Kevin", "Laura", "Brian", "Priya", "Carlos", "Wei", "Aisha"],
    "receiver": ["Mr. Smith", "Mrs. Garcia", "Ms. Lee", "Mr. Patel", "Mrs. Nguyen", "Mr. Walker"],
    "city": ["Texas", "Houston", "El Paso", "Miami", "Phoenix", "Chicago", "Atlanta"],
    "bank": ["Chase", "Wells Fargo", "Bank of America", "Citibank", "Capital One"],
    "clinic": ["Riverside Medical Center", "Dr. Alvarez's office", "Lakeview Family Clinic", "Sunrise Dental"],
    "pharmacy": ["CVS", "Walgreens"],
    "courier": ["FedEx", "UPS"],
    "utility": ["City Power and Light", "Metro Gas Company", "the electric company"],
    "solar": ["Sunshine Solar", "BrightPath Energy Solutions", "Evergreen Home Solar"],
    "insurer": ["State Farm", "Geico"],
    "school": ["Lincoln Elementary", "Westside High School"],
    "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "time": ["9:30", "10:15", "2:00", "3:45", "11:00"],
    "relative": ["grandson", "granddaughter", "nephew"],
    "item": ["an iPhone 15", "a MacBook", "a PlayStation 5", "a 65 inch TV"],
}

R_OPEN_A = ["Hello?", "Yes, speaking.", "Hi, who is this?", "Hello, how can I help you?"]
R_OPEN_B = ["Yeah, go ahead.", "Hi there.", "Speaking, what's this about?", "Mm-hm, who's calling?"]
R_WORRY_A = ["Oh no, what happened?", "That sounds serious.", "Really? I had no idea.", "Wait, what?"]
R_WORRY_B = ["Hmm, that's strange.", "Seriously? Since when?", "I'm not sure I follow.", "Okay, go on."]
R_DOUBT_A = ["Why do you need that?", "I don't feel comfortable sharing that.", "Can I call you back?",
             "Okay, I guess so."]
R_DOUBT_B = ["Is that really necessary?", "I'd rather not say that on the phone.", "Hmm, fine.",
             "Let me think about it."]
R_END_A = ["I'm going to hang up and check this myself.", "Alright, I'll do it now.", "Let me talk to my family first.",
           "Okay, thank you.", "No thanks, goodbye."]
R_END_B = ["I'll call them directly, bye.", "Fine, give me a minute.", "Not today, sorry.", "Alright, thanks.",
           "Sounds good."]

SCAMS = [
    Script("ssa_suspension", 1, "fraudster", [
        Step("C", ["Hello, this is Officer {agent} from the Social Security Administration.",
                   "Good afternoon, my name is Agent {agent} and I am calling from the Social Security Administration."],
             ["Hi, you are speaking with Special Agent {agent}, calling on behalf of the SSA.",
              "This is Investigator {agent} with the federal Social Security office."],
             [CL("government_agent"), ORG("Social Security Administration")]),
        Step("R", R_OPEN_A, R_OPEN_B),
        Step("C", [("Your Social Security number has been suspended due to suspicious activity in {city}.",
                    [TAC("threat", "suspended"), TAC("fear", "suspicious activity")]),
                   ("We found fraudulent activity connected to your Social Security number and it will be suspended.",
                    [TAC("threat", "suspended"), TAC("fear", "fraudulent activity")])],
             [("We detected illegal activity linked to your SSN and your benefits are going to be blocked.",
               [TAC("threat", "blocked"), TAC("fear", "illegal activity")]),
              ("A car rented under your SSN was found with illegal drugs in {city}, so your number is now frozen.",
               [TAC("threat", "frozen"), TAC("fear", "illegal drugs")])]),
        Step("R", R_WORRY_A, R_WORRY_B),
        Step("C", [("If you do not act today, a warrant will be issued for your arrest.",
                    [TAC("threat", "arrest"), TAC("urgency", "today")]),
                   ("You must resolve this right now or local police will arrest you within the hour.",
                    [TAC("threat", "arrest"), TAC("urgency", "right now")])],
             [("Unless this is settled immediately, officers will be sent to your home to arrest you.",
               [TAC("urgency", "immediately"), TAC("threat", "arrest")]),
              ("This is time sensitive, a lawsuit will be filed against you before the end of the day.",
               [TAC("urgency", "time sensitive"), TAC("threat", "lawsuit")])], p=0.9),
        Step("C", [("To protect your benefits I need you to verify your Social Security number and date of birth.",
                    [REQ("ssn"), REQ("personal_info", "date of birth")]),
                   ("Please confirm your Social Security number so I can open your case file.", [REQ("ssn")])],
             [("Please read me the full SSN on your card so I can confirm your identity.", [REQ("ssn")]),
              ("I need you to confirm your SSN and home address for the case file.",
               [REQ("ssn"), REQ("personal_info", "home address")])], p=0.85),
        Step("R", R_DOUBT_A, R_DOUBT_B),
        Step("C", [("You also have to pay a fine of {amount} using Google Play gift cards.",
                    [REQ("money", "fine"), PAY("gift_card", "Google Play")]),
                   ("To clear your record you must transfer your savings to a safe account today.",
                    [REQ("money", "savings"), PAY("bank_transfer", "safe account"), TAC("urgency", "today")]),
                   ("The penalty of {amount} must be paid in Bitcoin at a Bitcoin ATM near you.",
                    [REQ("money", "penalty"), PAY("crypto", "Bitcoin ATM")])],
             [("The fee of {amount} has to be paid with Target gift cards and you will read me the numbers.",
               [REQ("money", "fee"), PAY("gift_card", "Target"), ORG("Target")]),
              ("You need to withdraw {amount} in cash and a courier will pick it up from you.",
               [REQ("money"), PAY("cash", "courier pickup")]),
              ("Send the {amount} penalty through Western Union to our federal agent.",
               [REQ("money", "penalty"), PAY("wire_transfer", "Western Union"), ORG("Western Union")])], p=0.85),
        Step("C", [("Do not tell anyone about this, the case is confidential.", [TAC("secrecy", "confidential")])],
             [("Stay on the line with me and don't tell your bank why you are withdrawing.",
               [TAC("secrecy", "stay on the line")])], p=0.5),
        Step("C", [("If we get disconnected, call me back at {phone}.", [NUM()])],
             [("My direct number is {phone} in case the call drops.", [NUM()])], p=0.6),
        Step("R", R_END_A, R_END_B),
    ]),
    Script("irs_tax", 1, "fraudster", [
        Step("C", ["This is {first} {agent} from the Internal Revenue Service, badge number 4471.",
                   "Hello, I am calling from the IRS tax enforcement department."],
             ["Good morning, this is the IRS legal division, my name is {first}.",
              "I'm an agent with the Internal Revenue Service calling about your tax file."],
             [CL("government_agent"), ORG("Internal Revenue Service")]),
        Step("R", R_OPEN_A, R_OPEN_B),
        Step("C", [("Our records show you owe {amount} in back taxes and a lawsuit has been filed against you.",
                    [TAC("threat", "lawsuit"), REQ("money", "back taxes")]),
                   ("You owe {amount} in unpaid taxes and there is a warrant for your arrest.",
                    [TAC("threat", "arrest"), REQ("money", "unpaid taxes")])],
             [("There is a tax debt of {amount} on your file and legal action will start against you.",
               [TAC("threat", "legal action"), REQ("money", "tax debt")]),
              ("Because of an audit error you owe {amount}, and your bank account will be frozen.",
               [TAC("threat", "frozen"), REQ("money")])]),
        Step("R", R_WORRY_A, R_WORRY_B),
        Step("C", [("You must pay immediately to avoid jail.", [TAC("urgency", "immediately"), TAC("threat", "jail")]),
                   ("This must be paid within 24 hours or the police will come.", [TAC("urgency", "within 24 hours"), TAC("threat", "police")])],
             [("Payment is required right away, otherwise you face prosecution.",
               [TAC("urgency", "right away"), TAC("threat", "prosecution")])], p=0.9),
        Step("C", [("You can pay by wire transfer through MoneyGram.", [PAY("wire_transfer", "MoneyGram"), ORG("MoneyGram")]),
                   ("The only accepted payment is iTunes gift cards.", [PAY("gift_card", "iTunes")])],
             [("We accept payment in cryptocurrency, I will send you the wallet.", [PAY("crypto", "cryptocurrency")]),
              ("You can settle it with Google Play cards from any store.", [PAY("gift_card", "Google Play")])], p=0.85),
        Step("C", [("Also confirm your Social Security number for our records.", [REQ("ssn")])],
             [("Please verify your date of birth and full name.", [REQ("personal_info", "date of birth")])], p=0.5),
        Step("C", [("Do not hang up or discuss this with anyone.", [TAC("secrecy", "do not hang up")])],
             [("Keep this confidential, it is a federal matter.", [TAC("secrecy", "confidential")])], p=0.4),
        Step("R", R_END_A, R_END_B),
    ]),
    Script("tech_support", 1, "fraudster", [
        Step("C", ["Hello, this is {first} from Microsoft technical support.",
                   "Hi, I am calling from Windows support about your computer."],
             ["Good day, my name is {first}, I'm a certified Apple support technician.",
              "This is the Microsoft security team, I'm a senior technician."],
             [CL("tech_support")]),
        Step("R", R_OPEN_A, R_OPEN_B),
        Step("C", [("We detected a virus on your computer and your personal files have been hacked.",
                    [TAC("fear", "virus")]),
                   ("Your computer is sending error messages to our server, it is infected with malware.",
                    [TAC("fear", "malware")])],
             [("Hackers have compromised your device and your data is at risk.", [TAC("fear", "compromised")]),
              ("There is a security breach on your iCloud account.", [TAC("fear", "security breach"), ORG("Apple")])]),
        Step("R", R_WORRY_A, R_WORRY_B),
        Step("C", [("I need you to download a program called AnyDesk so I can fix it for you.",
                    [REQ("remote_access", "AnyDesk")]),
                   ("Please give me remote access to your computer so I can remove the virus.",
                    [REQ("remote_access")])],
             [("Go to our website and install this software so I can connect to your PC.",
               [REQ("remote_access", "install software")]),
              ("Let me get access to your laptop through TeamViewer.", [REQ("remote_access", "TeamViewer")])]),
        Step("C", [("It is urgent, you must act now before they steal everything.", [TAC("urgency", "act now")])],
             [("We need to do this immediately.", [TAC("urgency", "immediately")])], p=0.7),
        Step("C", [("The security license costs {amount}, please give me your credit card number.",
                    [REQ("money"), REQ("card_number"), PAY("credit_card_payment")]),
                   ("To renew your protection plan you need to pay {amount} with Steam cards.",
                    [REQ("money"), PAY("gift_card", "Steam")])],
             [("The repair fee is {amount}, you can pay with your debit card over the phone.",
               [REQ("money", "fee"), PAY("credit_card_payment")]),
              ("Log into your online banking so I can process the refund of {amount}.",
               [REQ("bank_account", "online banking"), TAC("reward", "refund")])], p=0.85),
        Step("R", R_DOUBT_A, R_DOUBT_B),
        Step("C", [("You can reach our support line at {phone}.", [NUM()])], [("Call back on {phone}, extension 204.", [NUM(), NUM("ext 204")])], p=0.5),
        Step("R", R_END_A, R_END_B),
    ]),
    Script("bank_otp", 1, "fraudster", [
        Step("C", ["Hello, this is {first} from the {bank} fraud department.",
                   "Good evening, I'm calling from {bank} card services."],
             ["Hi, I'm with the fraud prevention team at {bank}.",
              "This is the security department of your bank, {bank}."],
             [CL("bank"), ORG("{bank}")]),
        Step("R", R_OPEN_A, R_OPEN_B),
        Step("C", [("We noticed a suspicious transaction of {amount} on your account for {item}.",
                    [TAC("fear", "suspicious transaction")]),
                   ("Your account shows unauthorized charges from overseas.", [TAC("fear", "unauthorized charges")])],
             [("Someone tried to use your card for {item}, it looks like identity theft.", [TAC("fear", "identity theft")]),
              ("There are fraudulent charges on your debit card right now.", [TAC("fear", "fraudulent charges")])]),
        Step("R", R_WORRY_A, R_WORRY_B),
        Step("C", [("To block the transaction, please read me the verification code we just sent you.",
                    [REQ("otp", "verification code")]),
                   ("I will send you a one-time password, please confirm it to cancel the charge.",
                    [REQ("otp", "one-time password")])],
             [("You will get a six digit code by text, tell me that code so I can stop it.", [REQ("otp", "six digit code")]),
              ("Please give me the security code sent to your phone.", [REQ("otp", "security code")])]),
        Step("C", [("Also confirm your card number and CVV.", [REQ("card_number")]),
                   ("Please verify your online banking password.", [REQ("password")])],
             [("And I need the PIN for your debit card.", [REQ("password", "PIN")]),
              ("Could you confirm the account number and routing number?", [REQ("bank_account")])], p=0.6),
        Step("C", [("We must do this right now or your account will be frozen.", [TAC("urgency", "right now"), TAC("threat", "frozen")])],
             [("This is urgent, otherwise the account will be closed.", [TAC("urgency", "urgent"), TAC("threat", "account closure")])], p=0.6),
        Step("C", [("To keep your money safe, transfer your funds to a secure account I will give you by Zelle.",
                    [REQ("money"), PAY("zelle"), PAY("bank_transfer", "secure account")])],
             [("Move your savings into a safe account through Zelle and we will return it later.",
               [REQ("money", "savings"), PAY("zelle"), PAY("bank_transfer", "safe account")])], p=0.4),
        Step("C", [("Do not contact your branch, they are part of the investigation.", [TAC("secrecy", "do not contact")])],
             [("Don't call the bank, stay on the line with me.", [TAC("secrecy", "stay on the line")])], p=0.4),
        Step("R", R_END_A, R_END_B),
    ]),
    Script("prize_lottery", 1, "fraudster", [
        Step("C", ["Congratulations! This is {first} from Publishers Clearing House.",
                   "Hello, I'm calling from the national sweepstakes prize department."],
             ["Great news, I'm with the lottery commission and you are our winner.",
              "Hi, this is the prize patrol from Publishers Clearing House."],
             [CL("prize_or_lottery")]),
        Step("R", R_OPEN_A, R_OPEN_B),
        Step("C", [("You have won {amount},000 and a new car!", [TAC("reward", "won")]),
                   ("You've been selected to receive a cash prize of 2.5 million dollars.", [TAC("reward", "prize")])],
             [("Your number was drawn as the jackpot winner this month.", [TAC("reward", "jackpot")]),
              ("You are eligible to receive a free vacation and a cash reward.", [TAC("reward", "free vacation")])]),
        Step("R", ["Really? I never entered anything.", "Wow, are you serious?"], ["No way.", "That's hard to believe."]),
        Step("C", [("To release the prize you just need to pay a processing fee of {amount}.", [REQ("money", "processing fee")]),
                   ("You only have to pay the taxes on your winnings first, which is {amount}.", [REQ("money", "taxes")])],
             [("There is a small release fee of {amount} before we can deliver the prize.", [REQ("money", "release fee")])]),
        Step("C", [("You can send it through Western Union today.", [PAY("wire_transfer", "Western Union"), ORG("Western Union"), TAC("urgency", "today")]),
                   ("Please buy prepaid cards for the amount and read me the numbers.", [PAY("gift_card", "prepaid cards")])],
             [("Pay it with Cash App or Venmo to our claims agent.", [PAY("p2p_app", "Cash App")]),
              ("We accept a money order sent overnight.", [PAY("check_or_money_order", "money order")])], p=0.85),
        Step("C", [("We also need your bank account number to deposit the winnings.", [REQ("bank_account")])],
             [("Give me your account number and routing number for the deposit.", [REQ("bank_account")])], p=0.5),
        Step("C", [("This offer expires today, it's your last chance.", [TAC("urgency", "expires today")])],
             [("You must claim within 24 hours or the prize goes to someone else.", [TAC("urgency", "within 24 hours")])], p=0.6),
        Step("C", [("Keep this confidential until the prize is delivered.", [TAC("secrecy", "confidential")])],
             [("Don't tell anyone yet, it's a surprise announcement.", [TAC("secrecy", "don't tell anyone")])], p=0.4),
        Step("R", R_END_A, R_END_B),
    ]),
    Script("grandparent", 1, "fraudster", [
        Step("C", ["Grandma? It's me, your {relative}.", "Hi, it's me, your {relative}, I'm in big trouble."],
             ["Hey, it's your {relative}, please don't be mad.", "Grandpa, it's me, I need your help."],
             [CL("relative_or_friend", "{relative}")]),
        Step("R", ["Is that you? You sound different.", "Oh my goodness, what's wrong?"],
             ["What's going on, sweetheart?", "Are you okay?"]),
        Step("C", [("I was in a car accident and the police arrested me.", [TAC("fear", "accident"), TAC("threat", "arrested")]),
                   ("I'm in jail in {city} and I need bail money.", [TAC("threat", "jail"), REQ("money", "bail")])],
             [("There was an accident and I'm in the hospital, it's an emergency.", [TAC("fear", "emergency")]),
              ("I got into trouble abroad and they won't let me leave.", [TAC("fear", "in trouble")])]),
        Step("C", [("I need you to send {amount} for bail right now.", [REQ("money", "bail"), TAC("urgency", "right now")]),
                   ("Can you please send money immediately?", [REQ("money"), TAC("urgency", "immediately")])],
             [("Please help me pay {amount} as soon as possible.", [REQ("money"), TAC("urgency", "as soon as possible")])]),
        Step("C", [("A courier will come to pick up the cash at your house.", [PAY("cash", "courier pickup")]),
                   ("Please buy gift cards and read me the numbers.", [PAY("gift_card")])],
             [("You can wire the money through Western Union.", [PAY("wire_transfer", "Western Union"), ORG("Western Union")]),
              ("My lawyer can take the money in cash, he'll come by.", [PAY("cash")])], p=0.85),
        Step("C", [("Please don't tell Mom and Dad, they'll be so angry.", [TAC("secrecy", "don't tell")])],
             [("Keep this between us, okay? Don't tell anyone.", [TAC("secrecy", "keep this between us")])], p=0.8),
        Step("R", R_END_A, R_END_B),
    ]),
    Script("utility_disconnect", 1, "fraudster", [
        Step("C", ["Hello, this is {first} from {utility}, billing department.",
                   "This is the disconnection unit at {utility}."],
             ["Hi, I'm calling from your electric company about your account.",
              "Good afternoon, this is {utility} customer accounts."],
             [CL("utility_company")]),
        Step("R", R_OPEN_A, R_OPEN_B),
        Step("C", [("Your account is past due and your power will be shut off in thirty minutes.",
                    [TAC("threat", "shut off"), TAC("urgency", "thirty minutes")]),
                   ("Your service will be disconnected today because of an unpaid balance of {amount}.",
                    [TAC("threat", "disconnected"), TAC("urgency", "today"), REQ("money", "unpaid balance")])],
             [("A technician is on the way to cut off your electricity within the hour.",
               [TAC("threat", "cut off"), TAC("urgency", "within the hour")])]),
        Step("R", R_WORRY_A, R_WORRY_B),
        Step("C", [("To avoid the disconnection you must pay {amount} immediately.", [REQ("money"), TAC("urgency", "immediately")])],
             [("You need to pay the balance of {amount} right away.", [REQ("money"), TAC("urgency", "right away")])]),
        Step("C", [("You can pay through Zelle to our billing supervisor.", [PAY("zelle")]),
                   ("Please pay with a prepaid card from the pharmacy.", [PAY("gift_card", "prepaid card")])],
             [("We take payment by Cash App or Venmo only.", [PAY("p2p_app", "Cash App")]),
              ("Buy a gift card and call me back with the numbers.", [PAY("gift_card")])], p=0.9),
        Step("C", [("Call me back at {phone} once you have it.", [NUM()])], [("My direct line is {phone}.", [NUM()])], p=0.6),
        Step("R", R_END_A, R_END_B),
    ]),
    Script("amazon_refund", 1, "fraudster", [
        Step("C", ["Hello, this is {first} from Amazon customer service.",
                   "Hi, I'm calling from the Amazon security department about your order."],
             ["This is the Amazon refund department.",
              "Good day, I'm calling on behalf of Amazon about a charge on your account."],
             [CL("retailer_customer_service"), ORG("Amazon")]),
        Step("R", R_OPEN_A, R_OPEN_B),
        Step("C", [("There is an unauthorized order of {item} for {amount} on your account.", [TAC("fear", "unauthorized order")]),
                   ("Your account was hacked and someone ordered {item}.", [TAC("fear", "hacked")])],
             [("We see a suspicious purchase of {item} that we need to cancel.", [TAC("fear", "suspicious purchase")])]),
        Step("C", [("You are eligible for a refund of {amount}.", [TAC("reward", "refund")]),
                   ("We will issue you a full refund today.", [TAC("reward", "refund"), TAC("urgency", "today")])],
             [("I can process your refund right now.", [TAC("reward", "refund"), TAC("urgency", "right now")])], p=0.8),
        Step("C", [("To process it, please download the AnyDesk app so I can access your computer.",
                    [REQ("remote_access", "AnyDesk")]),
                   ("I need remote access to your computer to process the refund.", [REQ("remote_access")])],
             [("Install our support tool so I can see your screen, it's called TeamViewer.",
               [REQ("remote_access", "TeamViewer")])], p=0.8),
        Step("C", [("Oops, I refunded too much. You need to send back the difference with Amazon gift cards.",
                    [REQ("money"), PAY("gift_card", "Amazon gift cards")]),
                   ("Please log into your online banking so I can deposit the refund.", [REQ("bank_account", "online banking")])],
             [("We refunded {amount} by mistake, please return it with gift cards from Target.",
               [REQ("money"), PAY("gift_card", "Target"), ORG("Target")])], p=0.8),
        Step("R", R_DOUBT_A, R_DOUBT_B),
        Step("R", R_END_A, R_END_B),
    ]),
]

LEGIT = [
    Script("doctor_appointment", 0, "other_legitimate", [
        Step("C", ["Hi, this is {first} calling from {clinic}.", "Good morning, this is {clinic} calling for {receiver}."],
             ["Hello, it's {first} at {clinic}.", "Hi, this message is from {clinic}."],
             [CL("healthcare_provider"), ORG("{clinic}")]),
        Step("R", R_OPEN_A, R_OPEN_B),
        Step("C", ["I'm calling to remind you of your appointment on {day} at {time}.",
                   "We need to reschedule your checkup that was set for {day}."],
             ["Just a reminder about your visit with the doctor on {day} at {time}.",
              "The doctor asked me to follow up on your lab results, everything looks normal."]),
        Step("C", [("Can you confirm your date of birth for our records?", [REQ("personal_info", "date of birth")])],
             [("Could you verify your birth date please?", [REQ("personal_info", "date of birth")])], p=0.5),
        Step("R", ["Sure, that works for me.", "Yes, {day} is fine, thank you.", "Could we do a later time?"],
             ["Yes, I'll be there.", "Great, thanks for letting me know.", "Can we move it to the afternoon?"]),
        Step("C", ["Perfect, we'll see you then. Have a nice day!", "No problem, I've updated it. Take care!"],
             ["Wonderful, see you soon.", "Okay, you're all set."]),
    ]),
    Script("pharmacy_refill", 0, "other_legitimate", [
        Step("C", ["Hello, this is {first} from the {pharmacy} pharmacy.", "Hi, this is {pharmacy} pharmacy calling."],
             ["Hi, it's the pharmacist at {pharmacy}.", "Hello, this is {pharmacy} about your prescription."],
             [CL("healthcare_provider"), ORG("{pharmacy}")]),
        Step("R", R_OPEN_A, R_OPEN_B),
        Step("C", ["Your prescription refill is ready for pickup.", "Your medication will be ready after {time} today."],
             ["Your refill has been processed and you can pick it up anytime.",
              "We had to order your medication, it will arrive on {day}."]),
        Step("C", [("You can pay when you pick it up, your copay is 10 dollars.", [])],
             [("The copay stays the same as last time.", [])], p=0.5),
        Step("R", ["Great, thank you.", "I'll stop by later."], ["Thanks, I'll come get it.", "Perfect."]),
        Step("C", ["You're welcome, have a good day.", "Thanks, bye now."], ["See you soon.", "Take care."]),
    ]),
    Script("delivery_eta", 0, "other_legitimate", [
        Step("C", ["Hi, this is {first}, your driver from {courier}.", "Hello, I'm with {courier}, I have a package for you."],
             ["Hey, {courier} delivery here.", "Hi, it's the delivery driver from {courier}."],
             [CL("delivery_service"), ORG("{courier}")]),
        Step("R", R_OPEN_A, R_OPEN_B),
        Step("C", ["I'm about ten minutes away with your package.", "I'm outside but I can't find the building entrance."],
             ["Your parcel will be dropped off in about twenty minutes.", "I'm at the front gate, is there a code?"]),
        Step("C", [("Can you confirm the apartment number on your home address?", [REQ("personal_info", "home address")])],
             [("Just to double check, is your address the blue house on the corner?", [])], p=0.5),
        Step("R", ["Yes, it's apartment 4B. Thanks!", "Just leave it at the door, thank you."],
             ["The code is 1234, thanks.", "I'll come down now."]),
        Step("C", ["Great, see you soon.", "Okay, no problem."], ["Thanks, have a good one.", "Alright, dropping it off."]),
    ]),
    Script("bank_alert_legit", 0, "other_legitimate", [
        Step("C", ["Hello, this is {first} from {bank}.", "Hi, I'm calling from {bank} about a recent charge."],
             ["Good afternoon, this is the {bank} service center.", "Hello, {bank} calling for {receiver}."],
             [CL("bank"), ORG("{bank}")]),
        Step("R", R_OPEN_A, R_OPEN_B),
        Step("C", [("We noticed a suspicious charge of {amount} and wanted to check if you made it.",
                    [TAC("fear", "suspicious charge")]),
                   ("We saw a purchase at an electronics store, was that you?", [])],
             [("There's a transaction of {amount} that looks unusual, can you confirm it was yours?", [])]),
        Step("R", ["Yes, that was me, I bought a TV.", "No, that wasn't me."], ["Yes that's mine.", "Hmm, I don't recognize it."]),
        Step("C", ["Thank you. We will never ask for your PIN or password over the phone.",
                   "If you have concerns, please call the number on the back of your card."],
             ["Remember, we will never ask for your password or codes by phone.",
              "For your security, please call us using the number on your card."]),
        Step("R", R_END_A, R_END_B),
    ]),
    Script("friend_catchup", 0, "other_legitimate", [
        Step("C", ["Hey, it's {first}! Long time no talk.", "Hi, it's your old friend {first} from college."],
             ["Yo, it's {first}, got a minute?", "Hey you, it's {first}, how have you been?"],
             [CL("relative_or_friend", "friend")]),
        Step("R", ["Oh wow, hi! How are you?", "Hey! Good to hear from you."], ["Hey! What's up?", "No way, hi!"]),
        Step("C", ["I'm in town next week, do you want to grab dinner on {day}?",
                   "I just wanted to say happy birthday!"],
             ["Want to catch the game this weekend?", "I saw a photo of our old team and thought of you."]),
        Step("R", ["That sounds great, let's do it.", "Thanks so much!"], ["Sure, text me the details.", "Ha, that's awesome."]),
        Step("C", ["Awesome, I'll text you. Take care!", "Talk soon!"], ["Cool, see you then.", "Later!"]),
    ]),
    Script("sales_solar", 0, "salesperson", [
        Step("C", ["Hi, this is {first} with {solar}, how are you today?", "Good afternoon, I'm calling from {solar}."],
             ["Hey, {first} here from {solar}.", "Hello, this is {solar}, we're in your neighborhood this week."],
             [CL("telemarketer_sales"), ORG("{solar}")]),
        Step("R", R_OPEN_A, R_OPEN_B),
        Step("C", [("We have a special offer on solar panels that can cut your electric bill in half.", [TAC("reward", "special offer")]),
                   ("Homeowners in your area qualify for a discount on our installation.", [TAC("reward", "discount")])],
             [("We're running a promotion with a big discount for new customers.", [TAC("reward", "discount")])]),
        Step("C", [("It's a limited time offer, it ends this month.", [TAC("urgency", "limited time")]),
                   ("If you schedule a consultation you get a $50 gift card.", [TAC("reward", "gift card")])],
             [("The deal is only available until Friday.", [])], p=0.6),
        Step("R", ["I'm not interested, thanks.", "Hmm, send me some information."],
             ["Please take me off your list.", "Maybe, can you email me?"]),
        Step("C", [("Sure, what's the best email address to send it to?", [REQ("personal_info", "email address")]),
                   ("No problem, I'll remove your number. Have a great day.", [])],
             [("Of course, I'll mail you a brochure.", [])]),
    ]),
    Script("insurance_renewal", 0, "other_legitimate", [
        Step("C", ["Hi {receiver}, this is {first}, your agent at {insurer}.", "Hello, I'm calling from {insurer} about your policy."],
             ["Good morning, {insurer} calling about your car insurance.", "Hi, it's your {insurer} agent."],
             [CL("insurance"), ORG("{insurer}")]),
        Step("R", R_OPEN_A, R_OPEN_B),
        Step("C", ["Your auto policy renews next month and the rate stays the same.",
                   "I wanted to review your home insurance coverage before renewal."],
             ["Your policy is up for renewal and we found a lower premium for you.",
              "Just checking that your contact details are still correct."]),
        Step("C", [("Could you confirm the policy number on your card?", [REQ("personal_info", "policy number")])],
             [("You can review everything online whenever you like.", [])], p=0.5),
        Step("R", ["Okay, sounds good.", "Great, thanks for checking."], ["Perfect, thanks.", "I'll look it over."]),
        Step("C", ["You're welcome, call us anytime.", "Thanks, have a great day."], ["Take care.", "Bye now."]),
    ]),
    Script("prank_call", 0, "prankster", [
        Step("C", ["Hello, this is the President of the United States calling.", "Is your refrigerator running?"],
             ["Hi, I'm calling from the Department of Silly Walks.", "Hello, is this the pizza place?"]),
        Step("R", ["What? Who is this?", "Yes, why?"], ["Uh... no?", "Excuse me?"]),
        Step("C", ["Then you'd better go catch it! Ha ha!", "Gotcha! It's me, {first}, from next door!"],
             ["Just kidding! It's {first}, I got a new phone.", "Ha, I'm just messing with you, it's {first}."],
             [CL("relative_or_friend", "friend")]),
        Step("R", ["Very funny.", "Oh my god, you got me."], ["Not cool, man.", "Ha, you idiot."]),
    ]),
    Script("school_call", 0, "other_legitimate", [
        Step("C", ["Hello, this is {first} from {school} front office.", "Hi, I'm the nurse at {school}."],
             ["Good morning, {school} attendance office calling.", "Hi, this is your son's teacher at {school}."],
             [CL("other", "school"), ORG("{school}")]),
        Step("R", R_OPEN_A, R_OPEN_B),
        Step("C", ["Your son isn't feeling well, could someone pick him up today?",
                   "Just a reminder that the field trip form is due on {day}."],
             ["Your daughter was marked absent this morning, we're just checking in.",
              "Parent teacher conferences are on {day}, would {time} work for you?"]),
        Step("R", ["Thanks, I'll come right away.", "Okay, I'll sign it tonight."], ["She's home sick, sorry.", "Yes, that works."]),
        Step("C", ["Thank you, see you soon.", "Great, thanks!"], ["No problem, feel better.", "Perfect, thank you."]),
    ]),
    Script("job_interview", 0, "other_legitimate", [
        Step("C", ["Hi, this is {first} from human resources at Northwind Logistics.",
                   "Hello, I'm a recruiter with Contoso Staffing."],
             ["Hi, I'm the hiring manager at Fabrikam, calling about your application.",
              "Good morning, this is talent acquisition from Northwind Logistics."],
             [CL("employer_recruiter")]),
        Step("R", R_OPEN_A, R_OPEN_B),
        Step("C", ["We'd like to invite you for an interview on {day} at {time}.",
                   "Thanks for applying, are you still interested in the analyst position?"],
             ["We reviewed your resume and would love to set up a video interview.",
              "Are you available for a short phone screen on {day}?"]),
        Step("C", [("What's the best email address to send the calendar invite to?", [REQ("personal_info", "email address")])],
             [("I'll send the details to the email on your application.", [])], p=0.5),
        Step("R", ["Yes, that works, thank you!", "Could we do {day} instead?"], ["Sounds great.", "Sure, I'm available."]),
        Step("C", ["Great, we look forward to speaking with you.", "Perfect, talk to you then."],
             ["Excellent, see you then.", "Thanks, have a nice day."]),
    ]),
]

ROBOCALLS = [
    Script("robo_ssa", 1, "fraudster", [
        Step("C", ["This is an important message from the Social Security Administration.",
                   "This call is from the Department of Social Security."], None,
             [CL("government_agent"), ORG("Social Security Administration")]),
        Step("C", [("Your Social Security number has been suspended because of suspicious activity.",
                    [TAC("threat", "suspended"), TAC("fear", "suspicious activity")]),
                   ("Legal action has been filed against your Social Security number.", [TAC("threat", "legal action")])]),
        Step("C", [("To avoid arrest, press 1 now to speak with an officer.", [TAC("threat", "arrest"), NUM("press 1")]),
                   ("Call us back immediately at {phone} before the case is sent to the police.",
                    [TAC("urgency", "immediately"), NUM(), TAC("threat", "police")])]),
    ]),
    Script("robo_warranty", 1, "fraudster", [
        Step("C", ["We are calling about your car's extended warranty.", "This is the vehicle service department."], None,
             [CL("insurance")]),
        Step("C", [("This is your final notice, your warranty expires today.", [TAC("urgency", "expires today")]),
                   ("Your coverage is about to expire, this is your last chance to renew.", [TAC("urgency", "last chance")])]),
        Step("C", [("Press 1 to speak with a specialist and renew with your credit card.",
                    [NUM("press 1"), REQ("card_number"), PAY("credit_card_payment")]),
                   ("Press 1 now to keep your coverage.", [NUM("press 1")])]),
    ]),
    Script("robo_amazon", 1, "fraudster", [
        Step("C", ["This is Amazon security.", "Hello, this is an automated call from Amazon."], None,
             [CL("retailer_customer_service"), ORG("Amazon")]),
        Step("C", [("An order of {item} for {amount} was placed on your account and flagged as suspicious.",
                    [TAC("fear", "suspicious")]),
                   ("Your account has been compromised and will be suspended.", [TAC("fear", "compromised"), TAC("threat", "suspended")])]),
        Step("C", [("If you did not authorize this purchase, press 1 to talk to a representative.", [NUM("press 1")]),
                   ("Please call our fraud line at {phone} immediately.", [NUM(), TAC("urgency", "immediately")])]),
    ]),
    Script("robo_interest", 1, "fraudster", [
        Step("C", ["This is card member services calling about your credit card account.",
                   "Hello, this is your credit card services department."], None, [CL("bank")]),
        Step("C", [("You are eligible to lower your interest rate to zero percent.", [TAC("reward", "lower interest rate")]),
                   ("This is your final chance to reduce your interest rate.", [TAC("urgency", "final chance"), TAC("reward", "lower interest rate")])]),
        Step("C", [("Press 1 now and have your card number ready.", [NUM("press 1"), REQ("card_number")]),
                   ("Press 1 to speak to a specialist, this offer expires today.", [NUM("press 1"), TAC("urgency", "expires today")])]),
    ]),
]


def _pick(rng: np.random.Generator, pool: list):
    return pool[int(rng.integers(len(pool)))]


def _fmt(t: tuple, slots: dict) -> dict:
    rel, typ, obj, detail = t
    return {"relation": rel, "object_type": typ, "object": obj.format(**slots),
            "detail": detail.format(**slots), "inferred": False}


def render(script: Script, rng: np.random.Generator, style: str) -> tuple[str, list[dict]]:
    slots = {k: _pick(rng, v) for k, v in SLOTS.items()}
    slots["amount"] = f"${_pick(rng, [199, 250, 299, 480, 500, 750, 999, 1200, 1500, 2000])}"
    slots["phone"] = f"1-{_pick(rng, ['800', '888', '877', '866', '844'])}-{int(rng.integers(200, 999))}-{int(rng.integers(1000, 9999))}"
    turns: list[tuple[str, str]] = []
    triples: dict[tuple[str, str], dict] = {}
    for step in script.steps:
        if rng.random() > step.p:
            continue
        pool = step.b if (style == "B" and step.b) else step.a
        choice = _pick(rng, pool)
        text, extra = (choice, []) if isinstance(choice, str) else choice
        role = "caller" if step.role == "C" else "receiver"
        text = text.format(**slots)
        if turns and turns[-1][0] == role:
            turns[-1] = (role, turns[-1][1] + " " + text)
        else:
            turns.append((role, text))
        for t in list(step.triples) + list(extra):
            ft = _fmt(t, slots)
            triples.setdefault((ft["relation"], ft["object"]), ft)
    if style == "robo":
        transcript = " ".join(u for _, u in turns)
    elif style == "B":
        transcript = "\n".join(f"Speaker {1 if r == 'caller' else 2}: {u}" for r, u in turns)
    else:
        transcript = " ".join(f"{r}: {u}" for r, u in turns)
    return transcript, list(triples.values())


def _make(dataset: str, scripts: list[Script], n: int, style: str, rng: np.random.Generator,
          scam_share: float | None) -> list[dict]:
    scams = [s for s in scripts if s.label == 1]
    legit = [s for s in scripts if s.label == 0]
    out = []
    for i in range(n):
        if scam_share is None:
            script = _pick(rng, scripts)
        else:
            script = _pick(rng, scams if (i < round(n * scam_share)) else legit)
        text, gold = render(script, rng, style)
        out.append({"dataset": dataset, "source_id": f"{dataset}-{i:05d}", "raw_text": text, "label": script.label,
                    "label3": script.label3, "subtype": script.name, "gold_triples": gold})
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def generate_demo_data(out_dir: Path, n_primary: int = 600, n_shifted: int = 240, n_robocall: int = 120,
                       seed: int = 42) -> list[Path]:
    """Write demo_primary.jsonl, demo_shifted.jsonl and demo_robocall.jsonl into `out_dir`."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = {
        "demo_primary": _make("demo_primary", SCAMS + LEGIT, n_primary, "A", rng, 0.5),
        "demo_shifted": _make("demo_shifted", SCAMS + LEGIT, n_shifted, "B", rng, 0.5),
        "demo_robocall": _make("demo_robocall", ROBOCALLS, n_robocall, "robo", rng, None),
    }
    paths = []
    for name, recs in sets.items():
        p = out_dir / f"{name}.jsonl"
        write_jsonl(p, recs)
        paths.append(p)
        log.info("Synthetic %-14s %4d calls (scam rate %.2f) -> %s", name, len(recs),
                 np.mean([r["label"] for r in recs]), p)
    return paths
