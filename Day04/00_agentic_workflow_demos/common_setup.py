from langchain_core.documents import Document

# Rich enterprise-like documents (intentionally includes noise + overlaps)
documents = [

    # -------------------------
    # LEAVE POLICY
    # -------------------------
    Document(page_content="Employees are entitled to 20 days of paid annual leave per year."),
    Document(page_content="Employees can carry forward a maximum of 5 unused leave days to the next year."),
    Document(page_content="Sick leave entitlement is 10 days per year and cannot be carried forward."),
    Document(page_content="Maternity leave is provided for up to 26 weeks as per company policy."),
    Document(page_content="Paternity leave is granted for 10 days."),
    Document(page_content="Leave requests must be submitted at least 3 days in advance."),
    Document(page_content="Emergency leave can be applied retroactively with manager approval."),
    Document(page_content="Unpaid leave may be granted in exceptional circumstances."),
    
    # -------------------------
    # TRAVEL POLICY
    # -------------------------
    Document(page_content="Travel expenses are reimbursed up to $100 per day for domestic travel."),
    Document(page_content="International travel requires prior approval from senior management."),
    Document(page_content="Employees must book flights through the company travel portal."),
    Document(page_content="Hotel stays are capped at $150 per night for domestic travel."),
    Document(page_content="Taxi expenses are reimbursable with valid receipts."),
    Document(page_content="Meal expenses during travel are capped at $30 per meal."),
    Document(page_content="Luxury travel expenses will not be reimbursed."),
    Document(page_content="Travel claims must be submitted within 7 days of trip completion."),

    # -------------------------
    # EXPENSE & REIMBURSEMENT
    # -------------------------
    Document(page_content="All reimbursements require valid receipts and approval."),
    Document(page_content="Expense claims above $500 require finance team approval."),
    Document(page_content="Reimbursements are processed within 10 working days."),
    Document(page_content="Internet expenses for remote work are reimbursed up to $50 per month."),
    Document(page_content="Office supplies can be reimbursed up to $100 per month."),
    Document(page_content="Personal expenses are not eligible for reimbursement."),

    # -------------------------
    # WORK FROM HOME / HR POLICY
    # -------------------------
    Document(page_content="Employees are allowed to work from home up to 3 days per week."),
    Document(page_content="Work from home requests must be approved by the reporting manager."),
    Document(page_content="Employees must be available online during core working hours from 10 AM to 4 PM."),
    Document(page_content="Laptop and necessary equipment will be provided by the company."),
    Document(page_content="Employees must ensure a stable internet connection while working remotely."),

    # -------------------------
    # PERFORMANCE & BONUS
    # -------------------------
    Document(page_content="Performance reviews are conducted twice a year."),
    Document(page_content="Employees are eligible for annual bonuses based on performance ratings."),
    Document(page_content="Top performers may receive additional incentives."),
    Document(page_content="Performance improvement plans are initiated for underperforming employees."),

    # -------------------------
    # EDGE CASE / NOISE (IMPORTANT FOR DEMO)
    # -------------------------
    Document(page_content="The company cafeteria serves vegetarian and vegan meals."),
    Document(page_content="Office timings are from 9 AM to 6 PM."),
    Document(page_content="Employees must wear ID badges at all times inside office premises."),
]
