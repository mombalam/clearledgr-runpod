#!/usr/bin/env python3
"""
Prepare Clearledgr Financial Training Dataset

Creates a specialized financial training dataset for Llama 3.1 8B fine-tuning
with domain-specific examples for our AI finance agents.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset
import pandas as pd

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def create_financial_training_examples() -> List[Dict[str, str]]:
    """Create financial domain training examples for Clearledgr agents"""
    
    examples = []
    
    # Reconciliation examples
    reconciliation_examples = [
        {
            "instruction": "Reconcile the following bank statement with accounting records",
            "input": "Bank balance: $25,430.50. Outstanding checks: Check #1234 for $850.00, Check #1235 for $1,200.00. Deposits in transit: $2,100.00. Bank fees: $25.00. Interest earned: $15.50.",
            "output": "Book balance calculation: Bank balance $25,430.50 + Deposits in transit $2,100.00 - Outstanding checks ($850.00 + $1,200.00) - Bank fees $25.00 + Interest earned $15.50 = Book balance $25,471.00. Adjusting entries needed: 1) Debit Bank Fees Expense $25.00, Credit Cash $25.00. 2) Debit Cash $15.50, Credit Interest Income $15.50."
        },
        {
            "instruction": "Identify discrepancies in this bank reconciliation",
            "input": "Company records show cash balance of $18,500. Bank statement shows $19,200. Outstanding checks total $1,100. Deposits in transit: $900. No bank fees or interest.",
            "output": "Expected book balance: $19,200 - $1,100 + $900 = $19,000. Company records show $18,500, creating a $500 discrepancy. Investigation needed for: unrecorded bank transactions, posting errors, or timing differences. Recommend reviewing all transactions from the last business day."
        }
    ]
    
    # Financial reporting examples
    reporting_examples = [
        {
            "instruction": "Generate a cash flow statement from the following data",
            "input": "Net income: $150,000. Depreciation: $25,000. Accounts receivable increase: $10,000. Inventory decrease: $5,000. Accounts payable increase: $8,000. Equipment purchase: $50,000. Loan proceeds: $30,000. Dividends paid: $20,000.",
            "output": "Cash Flow Statement:\nOperating Activities:\nNet income: $150,000\nAdjustments: Depreciation $25,000\nChanges in working capital: AR increase ($10,000), Inventory decrease $5,000, AP increase $8,000\nNet operating cash flow: $178,000\n\nInvesting Activities:\nEquipment purchase: ($50,000)\nNet investing cash flow: ($50,000)\n\nFinancing Activities:\nLoan proceeds: $30,000\nDividends paid: ($20,000)\nNet financing cash flow: $10,000\n\nNet change in cash: $138,000"
        },
        {
            "instruction": "Calculate key financial ratios from balance sheet and income statement",
            "input": "Current assets: $500,000. Current liabilities: $200,000. Total assets: $1,200,000. Total equity: $800,000. Revenue: $2,000,000. Net income: $180,000. Cost of goods sold: $1,200,000.",
            "output": "Key Financial Ratios:\nLiquidity: Current ratio = $500,000 ÷ $200,000 = 2.5\nLeverage: Debt-to-equity = ($1,200,000 - $800,000) ÷ $800,000 = 0.5\nProfitability: Net margin = $180,000 ÷ $2,000,000 = 9%\nEfficiency: Asset turnover = $2,000,000 ÷ $1,200,000 = 1.67\nProfitability: ROE = $180,000 ÷ $800,000 = 22.5%\nGross margin = ($2,000,000 - $1,200,000) ÷ $2,000,000 = 40%"
        }
    ]
    
    # Journal entry examples
    journal_examples = [
        {
            "instruction": "Create journal entries for the following transactions",
            "input": "1) Purchased office supplies for $500 cash. 2) Received $10,000 from customer for services to be performed next month. 3) Paid monthly rent of $2,000. 4) Recorded depreciation expense of $1,500.",
            "output": "Journal Entries:\n1) Dr. Office Supplies $500\n   Cr. Cash $500\n2) Dr. Cash $10,000\n   Cr. Unearned Revenue $10,000\n3) Dr. Rent Expense $2,000\n   Cr. Cash $2,000\n4) Dr. Depreciation Expense $1,500\n   Cr. Accumulated Depreciation $1,500"
        },
        {
            "instruction": "Record adjusting entries at month-end",
            "input": "Insurance policy paid $12,000 for 12 months on January 1st. It's now January 31st. Accrued wages of $3,500. Supplies on hand $800, supplies account shows $1,200.",
            "output": "Adjusting Entries:\n1) Insurance expense adjustment:\n   Dr. Insurance Expense $1,000\n   Cr. Prepaid Insurance $1,000\n   (1 month: $12,000 ÷ 12 = $1,000)\n\n2) Accrued wages:\n   Dr. Wages Expense $3,500\n   Cr. Wages Payable $3,500\n\n3) Supplies used:\n   Dr. Supplies Expense $400\n   Cr. Supplies $400\n   (Used: $1,200 - $800 = $400)"
        }
    ]
    
    # Budgeting and variance analysis
    budget_examples = [
        {
            "instruction": "Perform variance analysis on the budget vs actual results",
            "input": "Budget: Revenue $100,000, Expenses $75,000. Actual: Revenue $95,000, Expenses $78,000. Analyze variances and provide explanations.",
            "output": "Variance Analysis:\nRevenue Variance: $95,000 - $100,000 = ($5,000) Unfavorable (5% below budget)\nExpense Variance: $78,000 - $75,000 = $3,000 Unfavorable (4% over budget)\nNet Income Impact: Budget NI $25,000 vs Actual NI $17,000 = ($8,000) unfavorable\n\nRecommendations:\n1) Investigate revenue shortfall - market conditions, sales process, pricing\n2) Review expense overruns - cost controls, unexpected expenses\n3) Adjust forecasts for remaining periods\n4) Implement corrective actions for revenue generation"
        }
    ]
    
    # Compliance and audit examples
    compliance_examples = [
        {
            "instruction": "Identify SOX compliance requirements for this financial process",
            "input": "Monthly financial close process including journal entries, account reconciliations, and financial statement preparation.",
            "output": "SOX Compliance Requirements:\n1) Documentation: Maintain complete audit trail for all journal entries\n2) Authorization: Require manager approval for entries over materiality threshold\n3) Segregation of duties: Separate preparation, review, and approval functions\n4) Account reconciliations: Independent review and sign-off required\n5) Supporting documentation: Retain all source documents and approvals\n6) Management certification: CFO/CEO sign-off on accuracy and completeness\n7) Internal controls testing: Document and test key controls quarterly\n8) IT controls: Access controls, change management, data backup procedures"
        }
    ]
    
    # Combine all examples
    all_examples = (
        reconciliation_examples + 
        reporting_examples + 
        journal_examples + 
        budget_examples + 
        compliance_examples
    )
    
    # Format for training
    formatted_examples = []
    for example in all_examples:
        # Create conversation format
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant specialized in financial operations and accounting. You provide accurate, detailed, and professional financial analysis and recommendations.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction']}\n\n{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{example['output']}<|eot_id|>"
        
        formatted_examples.append({"text": text})
    
    return formatted_examples

def create_agent_specific_examples() -> List[Dict[str, str]]:
    """Create examples for specific Clearledgr agents"""
    
    agent_examples = []
    
    # ReconciliationAgent examples
    recon_examples = [
        {
            "agent": "ReconciliationAgent",
            "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are ReconciliationAgent, a specialized AI for bank and account reconciliations. You excel at identifying discrepancies, timing differences, and creating adjustment entries.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nProcess this bank reconciliation: Bank shows $15,432.10. Books show $14,890.25. Outstanding checks: $1,245.60. Deposits in transit: $703.75.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nReconciliation Analysis:\n\nAdjusted Bank Balance:\nBank statement balance: $15,432.10\nLess: Outstanding checks: ($1,245.60)\nAdd: Deposits in transit: $703.75\nAdjusted bank balance: $14,890.25\n\nBook Balance: $14,890.25\n\nStatus: ✅ RECONCILED\nDifference: $0.00\n\nNo adjusting entries required. All differences are normal timing items that will clear in the next banking cycle.<|eot_id|>"
        }
    ]
    
    # DocsBot examples
    docs_examples = [
        {
            "agent": "DocsBot",
            "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are DocsBot, an AI specialist in document processing and data extraction. You accurately extract financial data from invoices, receipts, contracts, and statements.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nExtract key data from this invoice: Invoice #INV-2024-001, Date: 01/15/2024, Vendor: Office Supplies Co., Items: Paper $45.00, Pens $23.50, Folders $18.75, Subtotal: $87.25, Tax: $7.85, Total: $95.10, Due Date: 02/14/2024<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nDocument Extraction Results:\n\nInvoice Details:\n- Invoice Number: INV-2024-001\n- Invoice Date: 01/15/2024\n- Vendor: Office Supplies Co.\n- Due Date: 02/14/2024\n\nLine Items:\n1. Paper - $45.00\n2. Pens - $23.50  \n3. Folders - $18.75\n\nFinancial Summary:\n- Subtotal: $87.25\n- Tax: $7.85\n- Total Amount: $95.10\n\nSuggested Journal Entry:\nDr. Office Supplies Expense $87.25\nDr. Sales Tax Expense $7.85\nCr. Accounts Payable - Office Supplies Co. $95.10\n\nPayment Terms: Net 30 days<|eot_id|>"
        }
    ]
    
    # ReportGen examples
    report_examples = [
        {
            "agent": "ReportGen",
            "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are ReportGen, an AI specialist in financial reporting and analysis. You create professional financial statements, management reports, and executive dashboards.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGenerate a monthly P&L summary from this data: Revenue $125,000, COGS $75,000, Salaries $25,000, Rent $5,000, Utilities $2,000, Marketing $8,000, Other expenses $3,500<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nMONTHLY PROFIT & LOSS STATEMENT\n\nREVENUE\nTotal Revenue                     $125,000\n\nCOST OF GOODS SOLD\nCost of Goods Sold                 $75,000\nGross Profit                       $50,000\nGross Margin                         40.0%\n\nOPERATING EXPENSES\nSalaries & Benefits                $25,000\nRent Expense                        $5,000\nUtilities                           $2,000\nMarketing                           $8,000\nOther Operating Expenses            $3,500\nTotal Operating Expenses           $43,500\n\nNET INCOME                          $6,500\nNet Margin                            5.2%\n\nKEY METRICS:\n- Gross Margin: 40.0% (Healthy)\n- Operating Margin: 5.2% (Monitor)\n- Revenue/Employee: $5,000 (assuming 25 employees)\n\nRECOMMENDations:\n- Gross margin is strong at 40%\n- Consider reviewing marketing ROI\n- Monitor other expenses for optimization opportunities<|eot_id|>"
        }
    ]
    
    return recon_examples + docs_examples + report_examples

def main():
    logger = setup_logging()
    
    logger.info("🔄 Preparing Clearledgr financial training dataset...")
    
    # Create output directory
    output_dir = Path("./data/financial_training_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training examples
    logger.info("Generating general financial examples...")
    general_examples = create_financial_training_examples()
    
    logger.info("Generating agent-specific examples...")
    agent_examples = create_agent_specific_examples()
    
    # Combine all examples
    all_examples = general_examples + agent_examples
    
    logger.info(f"Created {len(all_examples)} training examples")
    
    # Create dataset
    dataset = Dataset.from_list(all_examples)
    
    # Save dataset
    dataset_path = output_dir / "clearledgr_financial_dataset"
    dataset.save_to_disk(str(dataset_path))
    
    # Also save as JSON for inspection
    json_path = output_dir / "examples.json"
    with open(json_path, 'w') as f:
        json.dump(all_examples, f, indent=2)
    
    # Create summary
    summary = {
        "dataset_info": {
            "total_examples": len(all_examples),
            "general_examples": len(general_examples),
            "agent_examples": len(agent_examples),
            "created_date": "2024-01-15",
            "format": "Llama 3.1 chat format",
            "purpose": "Fine-tuning for Clearledgr financial AI agents"
        },
        "categories": {
            "reconciliation": "Bank and account reconciliation examples",
            "reporting": "Financial statement and analysis examples", 
            "journal_entries": "Accounting transaction examples",
            "budgeting": "Budget and variance analysis examples",
            "compliance": "SOX and audit compliance examples",
            "agent_specific": "Examples tailored for specific Clearledgr agents"
        }
    }
    
    with open(output_dir / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Dataset saved to {dataset_path}")
    logger.info(f"📊 Dataset summary: {len(all_examples)} examples")
    logger.info(f"📁 JSON export: {json_path}")
    logger.info("🎯 Dataset ready for fine-tuning!")

if __name__ == "__main__":
    main()
