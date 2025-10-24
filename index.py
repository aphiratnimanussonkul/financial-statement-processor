"""
Financial Statement PDF Processor for Thai SME (DBD Format)
Extracts Balance Sheet and Profit & Loss Statement into structured JSON

Requirements:
pip install pdfplumber anthropic python-dotenv pandas pydantic

Usage:
python process_financial_statement.py input.pdf --output output.json
"""

import pdfplumber
import anthropic
import json
import re
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True  # Override any existing configuration
    )
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)


# ============================================================================
# STEP 1: SCHEMA DEFINITIONS
# ============================================================================


class CurrentAssets(BaseModel):
    """สินทรัพย์หมุนเวียน"""

    cash_and_equivalents: Optional[float] = Field(
        None, description="เงินสดและรายการเทียบเท่าเงินสด"
    )
    short_term_investments: Optional[float] = Field(None, description="เงินลงทุนระยะสั้น")
    accounts_receivable: Optional[float] = Field(None, description="ลูกหนี้การค้า")
    inventory: Optional[float] = Field(None, description="สินค้าคงเหลือ")
    other_current_assets: Optional[float] = Field(None, description="สินทรัพย์หมุนเวียนอื่น")
    total: Optional[float] = Field(None, description="รวมสินทรัพย์หมุนเวียน")


class NonCurrentAssets(BaseModel):
    """สินทรัพย์ไม่หมุนเวียน"""

    property_plant_equipment: Optional[float] = Field(
        None, description="ที่ดิน อาคาร และอุปกรณ์"
    )
    intangible_assets: Optional[float] = Field(None, description="สินทรัพย์ไม่มีตัวตน")
    long_term_investments: Optional[float] = Field(None, description="เงินลงทุนระยะยาว")
    other_non_current_assets: Optional[float] = Field(
        None, description="สินทรัพย์ไม่หมุนเวียนอื่น"
    )
    total: Optional[float] = Field(None, description="รวมสินทรัพย์ไม่หมุนเวียน")


class Assets(BaseModel):
    """สินทรัพย์"""

    current_assets: CurrentAssets
    non_current_assets: NonCurrentAssets
    total_assets: Optional[float] = Field(None, description="รวมสินทรัพย์")


class CurrentLiabilities(BaseModel):
    """หนี้สินหมุนเวียน"""

    accounts_payable: Optional[float] = Field(None, description="เจ้าหนี้การค้า")
    short_term_loans: Optional[float] = Field(None, description="เงินกู้ยืมระยะสั้น")
    other_current_liabilities: Optional[float] = Field(
        None, description="หนี้สินหมุนเวียนอื่น"
    )
    total: Optional[float] = Field(None, description="รวมหนี้สินหมุนเวียน")


class NonCurrentLiabilities(BaseModel):
    """หนี้สินไม่หมุนเวียน"""

    long_term_loans: Optional[float] = Field(None, description="เงินกู้ยืมระยะยาว")
    other_non_current_liabilities: Optional[float] = Field(
        None, description="หนี้สินไม่หมุนเวียนอื่น"
    )
    total: Optional[float] = Field(None, description="รวมหนี้สินไม่หมุนเวียน")


class Liabilities(BaseModel):
    """หนี้สิน"""

    current_liabilities: CurrentLiabilities
    non_current_liabilities: NonCurrentLiabilities
    total_liabilities: Optional[float] = Field(None, description="รวมหนี้สิน")


class Equity(BaseModel):
    """ส่วนของผู้ถือหุ้น"""

    share_capital: Optional[float] = Field(None, description="ทุนจดทะเบียน")
    retained_earnings: Optional[float] = Field(None, description="กำไรสะสม")
    other_equity: Optional[float] = Field(None, description="ส่วนของผู้ถือหุ้นอื่น")
    total_equity: Optional[float] = Field(None, description="รวมส่วนของผู้ถือหุ้น")


class BalanceSheet(BaseModel):
    """งบดุล / Balance Sheet"""

    assets: Assets
    liabilities: Liabilities
    equity: Equity

    @validator("equity")
    def check_balance_equation(cls, v, values):
        """ตรวจสอบสมการงบดุล: สินทรัพย์ = หนี้สิน + ส่วนของผู้ถือหุ้น"""
        if "assets" in values and "liabilities" in values:
            total_assets = values["assets"].total_assets or 0
            total_liabilities = values["liabilities"].total_liabilities or 0
            total_equity = v.total_equity or 0

            if abs(total_assets - (total_liabilities + total_equity)) > 100:
                logger.warning(
                    f"Balance sheet doesn't balance: {total_assets} ≠ {total_liabilities + total_equity}"
                )
        return v


class OperatingExpenses(BaseModel):
    """ค่าใช้จ่ายในการดำเนินงาน"""

    selling_expenses: Optional[float] = Field(None, description="ค่าใช้จ่ายในการขาย")
    administrative_expenses: Optional[float] = Field(
        None, description="ค่าใช้จ่ายในการบริหาร"
    )
    other_expenses: Optional[float] = Field(None, description="ค่าใช้จ่ายอื่น")
    total: Optional[float] = Field(None, description="รวมค่าใช้จ่ายในการดำเนินงาน")


class ProfitAndLoss(BaseModel):
    """งบกำไรขาดทุน / Profit & Loss Statement"""

    revenue: Optional[float] = Field(None, description="รายได้จากการขาย")
    cost_of_goods_sold: Optional[float] = Field(None, description="ต้นทุนขาย")
    gross_profit: Optional[float] = Field(None, description="กำไรขั้นต้น")
    operating_expenses: OperatingExpenses
    operating_profit: Optional[float] = Field(None, description="กำไรจากการดำเนินงาน")
    other_income: Optional[float] = Field(None, description="รายได้อื่น")
    other_expenses: Optional[float] = Field(None, description="ค่าใช้จ่ายอื่น")
    profit_before_tax: Optional[float] = Field(None, description="กำไรก่อนภาษี")
    tax_expense: Optional[float] = Field(None, description="ค่าใช้จ่ายภาษี")
    net_profit: Optional[float] = Field(None, description="กำไรสุทธิ")


class CompanyInfo(BaseModel):
    """ข้อมูลบริษัท"""

    name: Optional[str] = Field(None, description="ชื่อบริษัท")
    period_start: Optional[str] = Field(None, description="วันที่เริ่มต้นรอบบัญชี")
    period_end: Optional[str] = Field(None, description="วันที่สิ้นสุดรอบบัญชี")
    report_type: Optional[str] = Field(None, description="ประเภทรายงาน")
    currency: Optional[str] = Field("THB", description="สกุลเงิน")


class FinancialStatement(BaseModel):
    """งบการเงินฉบับสมบูรณ์"""

    company_info: CompanyInfo
    balance_sheet: Optional[BalanceSheet] = None
    profit_and_loss: Optional[ProfitAndLoss] = None
    extracted_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# STEP 2: TERM MAPPING DICTIONARY
# ============================================================================

TERM_MAPPING = """
## งบดุล (Balance Sheet)

### สินทรัพย์หมุนเวียน (Current Assets):
- เงินสด, เงินสดและเงินฝากธนาคาร, Cash, Cash and bank = cash_and_equivalents
- ลูกหนี้, ลูกหนี้การค้า, ลูกหนี้การค้าและลูกหนี้อื่น, Trade receivables, Accounts receivable = accounts_receivable
- สินค้า, สินค้าคงเหลือ, สต็อก, Inventory, Stock = inventory
- เงินลงทุนชั่วคราว, เงินลงทุนระยะสั้น, Short-term investments = short_term_investments

### สินทรัพย์ไม่หมุนเวียน (Non-Current Assets):
- ที่ดิน อาคาร อุปกรณ์, ที่ดิน อาคารและอุปกรณ์, ทรัพย์สินถาวร, PPE, Property Plant Equipment, Fixed assets = property_plant_equipment
- สินทรัพย์ไม่มีตัวตน, Intangible assets = intangible_assets
- เงินลงทุนระยะยาว, Long-term investments = long_term_investments

### หนี้สินหมุนเวียน (Current Liabilities):
- เจ้าหนี้, เจ้าหนี้การค้า, เจ้าหนี้การค้าและเจ้าหนี้อื่น, Trade payables, Accounts payable = accounts_payable
- เงินกู้ยืมระยะสั้น, เงินเบิกเกินบัญชี, หนี้สินระยะสั้น, Short-term loans = short_term_loans

### หนี้สินไม่หมุนเวียน (Non-Current Liabilities):
- เงินกู้ยืมระยะยาว, หนี้สินระยะยาว, Long-term loans, Long-term debt = long_term_loans

### ส่วนของผู้ถือหุ้น (Equity):
- ทุนจดทะเบียน, ทุน, Share capital, Capital = share_capital
- กำไรสะสม, กำไร(ขาดทุน)สะสม, Retained earnings = retained_earnings

## งบกำไรขาดทุน (Profit & Loss)

### รายได้และต้นทุน:
- รายได้, รายได้จากการขาย, รายได้จากการขายและบริการ, ยอดขาย, Sales, Revenue = revenue
- ต้นทุน, ต้นทุนขาย, ราคาทุนขาย, ต้นทุนของสินค้าที่ขาย, COGS, Cost of sales = cost_of_goods_sold
- กำไรขั้นต้น, Gross profit, GP = gross_profit

### ค่าใช้จ่าย:
- ค่าใช้จ่ายขาย, ค่าใช้จ่ายในการขาย, Selling expenses = selling_expenses
- ค่าใช้จ่ายบริหาร, ค่าใช้จ่ายในการบริหาร, Administrative expenses, Admin expenses = administrative_expenses
- กำไรจากการดำเนินงาน, กำไรจากการปฏิบัติงาน, Operating profit, EBIT = operating_profit

### กำไร:
- กำไรก่อนภาษี, Profit before tax, PBT, EBT = profit_before_tax
- ค่าใช้จ่ายภาษี, ภาษีเงินได้, Tax expense, Income tax = tax_expense
- กำไรสุทธิ, กำไร(ขาดทุน)สุทธิ, Net profit, Net income = net_profit
"""


# ============================================================================
# STEP 3: PDF PROCESSOR
# ============================================================================


class PDFProcessor:
    """ประมวลผล PDF และ extract ข้อความ"""

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    def extract_all_pages(self) -> List[Dict[str, Any]]:
        """Extract text จากทุกหน้า"""
        pages_data = []

        with pdfplumber.open(self.pdf_path) as pdf:
            logger.info(f"Processing {len(pdf.pages)} pages...")

            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                tables = page.extract_tables()

                pages_data.append(
                    {
                        "page_number": page_num,
                        "text": text,
                        "tables": tables,
                        "char_count": len(text),
                    }
                )

                logger.debug(
                    f"Page {page_num}: {len(text)} characters, {len(tables)} tables"
                )

        return pages_data

    def extract_complete_financial_statement(
        self, pages_data: List[Dict], client: anthropic.Anthropic
    ) -> Dict[str, Any]:
        """ใช้ Claude ในการ extract งบการเงินทั้งหมดในครั้งเดียว"""

        # สร้าง summary ของแต่ละหน้า
        pages_summary = []
        for page in pages_data[:20]:  # จำกัดแค่ 20 หน้าแรก
            preview = page["text"][:800]  # เพิ่มเป็น 800 ตัวอักษรเพื่อให้ข้อมูลครบถ้วน
            pages_summary.append(f"Page {page['page_number']}:\n{preview}\n{'='*50}")

        combined_summary = "\n\n".join(pages_summary)

        # สร้าง schema สำหรับผลลัพธ์ที่ต้องการ
        balance_sheet_schema = BalanceSheet.schema_json(indent=2)
        profit_and_loss_schema = ProfitAndLoss.schema_json(indent=2)
        company_info_schema = CompanyInfo.schema_json(indent=2)

        prompt = f"""คุณเป็นผู้เชี่ยวชาญด้านงบการเงินไทย วิเคราะห์เอกสารงบการเงินต่อไปนี้และ extract ข้อมูลทั้งหมดในครั้งเดียว
1. งบดุล (Balance Sheet) - มักมีคำว่า "งบฐานะการเงิน", "งบดุล", "สินทรัพย์", "หนี้สิน", "ส่วนของผู้ถือหุ้น"
2. งบกำไรขาดทุน (Profit & Loss) - มักมีคำว่า "งบกำไรขาดทุน", "งบกำไรขาดทุน", "รายได้", "ต้นทุนขาย", "กำไร", "ขาดทุน"

หน้าต่างๆ:


เอกสารงบการเงิน:
{combined_summary}

คำศัพท์ที่ใช้ในการ mapping:
{TERM_MAPPING}

กรุณาตอบเป็น JSON format ตามโครงสร้างนี้:
{{
    "company_info": {company_info_schema},
    "balance_sheet": {balance_sheet_schema},
    "profit_and_loss": {profit_and_loss_schema},
    "identified_pages": {{
        "balance_sheet_pages": [หมายเลขหน้าที่เป็นงบดุล],
        "profit_and_loss_pages": [หมายเลขหน้าที่เป็นงบกำไรขาดทุน]
    }},
    "reasoning": "อธิบายสั้นๆ ว่าหน้าไหนเป็นอะไร"
}}

หลักการสำคัญ:
1. จับคู่รายการที่มีความหมายเดียวกันแม้ชื่อเขียนต่างกัน (ใช้ TERM_MAPPING)
2. แปลงตัวเลขทั้งหมดเป็น float (ถ้าเป็นหน่วยพัน ให้คูณ 1000, ถ้าเป็นหน่วยล้าน ให้คูณ 1,000,000)
3. ถ้าไม่พบข้อมูลให้ใส่ null
4. ตรวจสอบความสมดุลของงบดุล: total_assets = total_liabilities + total_equity
5. ตรวจสอบกำไรขั้นต้น: gross_profit = revenue - cost_of_goods_sold
6. ตอบเป็น JSON เท่านั้น ไม่ต้องอธิบาย

JSON:"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=8000,  # เพิ่ม max_tokens เพื่อรองรับข้อมูลทั้งหมด
                messages=[{"role": "user", "content": prompt}],
            )

            result_text = response.content[0].text
            # Parse JSON from response
            json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                logger.info(f"Successfully extracted complete financial statement")
                return result
            else:
                logger.error("Could not find JSON in response")
                return None

        except Exception as e:
            logger.error(f"Error extracting complete financial statement: {e}")
            return None


# ============================================================================
# STEP 4: LLM EXTRACTOR
# ============================================================================


class FinancialExtractor:
    """ใช้ Claude API ในการ extract และ classify ข้อมูล - ใช้ API call เดียว"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)


# ============================================================================
# STEP 5: VALIDATOR
# ============================================================================


class FinancialValidator:
    """ตรวจสอบความถูกต้องของข้อมูล"""

    @staticmethod
    def validate_balance_sheet(bs: BalanceSheet) -> List[str]:
        """ตรวจสอบงบดุล"""
        issues = []

        # Check balance equation
        total_assets = bs.assets.total_assets or 0
        total_liabilities = bs.liabilities.total_liabilities or 0
        total_equity = bs.equity.total_equity or 0

        if abs(total_assets - (total_liabilities + total_equity)) > 100:
            diff = total_assets - (total_liabilities + total_equity)
            issues.append(
                f"งบดุลไม่สมดุล: สินทรัพย์ {total_assets:,.0f} ≠ หนี้สิน+ส่วนผู้ถือหุ้น {total_liabilities + total_equity:,.0f} (ต่าง {diff:,.0f})"
            )

        # Check current assets sum
        ca = bs.assets.current_assets
        ca_sum = sum(
            filter(
                None,
                [
                    ca.cash_and_equivalents,
                    ca.short_term_investments,
                    ca.accounts_receivable,
                    ca.inventory,
                    ca.other_current_assets,
                ],
            )
        )

        if ca.total and abs(ca_sum - ca.total) > 100:
            issues.append(f"สินทรัพย์หมุนเวียนรวมไม่ตรง: {ca_sum:,.0f} ≠ {ca.total:,.0f}")

        return issues

    @staticmethod
    def validate_profit_and_loss(pl: ProfitAndLoss) -> List[str]:
        """ตรวจสอบงบกำไรขาดทุน"""
        issues = []

        # Check gross profit
        revenue = pl.revenue or 0
        cogs = pl.cost_of_goods_sold or 0
        gross_profit = pl.gross_profit or 0

        if abs(gross_profit - (revenue - cogs)) > 100:
            issues.append(
                f"กำไรขั้นต้นไม่ตรง: {gross_profit:,.0f} ≠ {revenue:,.0f} - {cogs:,.0f}"
            )

        # Check operating profit
        if pl.operating_profit and pl.operating_expenses.total:
            expected_op = gross_profit - pl.operating_expenses.total
            if abs(pl.operating_profit - expected_op) > 100:
                issues.append(f"กำไรจากการดำเนินงานไม่ตรง")

        return issues


# ============================================================================
# STEP 6: MAIN PIPELINE
# ============================================================================


def process_financial_statement(
    pdf_path: str, anthropic_api_key: str, output_path: Optional[str] = None
) -> FinancialStatement:
    """Main pipeline สำหรับประมวลผลงบการเงิน - ใช้ API call เดียว"""

    logger.info(f"Processing: {pdf_path}")

    # Initialize components
    pdf_processor = PDFProcessor(pdf_path)
    extractor = FinancialExtractor(anthropic_api_key)
    validator = FinancialValidator()

    # Step 1: Extract all pages
    logger.info("Step 1: Extracting PDF pages...")
    pages_data = pdf_processor.extract_all_pages()

    # Step 2: Extract complete financial statement in one API call
    logger.info("Step 2: Extracting complete financial statement (single API call)...")
    extraction_result = pdf_processor.extract_complete_financial_statement(
        pages_data, extractor.client
    )

    if not extraction_result:
        logger.error("Failed to extract financial statement")
        return FinancialStatement(
            company_info=CompanyInfo(),
            metadata={"source_file": str(pdf_path), "total_pages": len(pages_data)}
        )

    # Parse the results
    company_info_data = extraction_result.get("company_info", {})
    balance_sheet_data = extraction_result.get("balance_sheet", {})
    profit_and_loss_data = extraction_result.get("profit_and_loss", {})
    identified_pages = extraction_result.get("identified_pages", {})
    reasoning = extraction_result.get("reasoning", "")

    logger.info(f"Extraction reasoning: {reasoning}")

    # Create Pydantic objects
    try:
        company_info = CompanyInfo(**company_info_data) if company_info_data else CompanyInfo()
    except Exception as e:
        logger.warning(f"Error parsing company info: {e}")
        company_info = CompanyInfo()

    balance_sheet = None
    if balance_sheet_data:
        try:
            balance_sheet = BalanceSheet(**balance_sheet_data)
            # Validate balance sheet
            issues = validator.validate_balance_sheet(balance_sheet)
            if issues:
                logger.warning("Balance Sheet validation issues:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
        except Exception as e:
            logger.warning(f"Error parsing balance sheet: {e}")

    profit_and_loss = None
    if profit_and_loss_data:
        try:
            profit_and_loss = ProfitAndLoss(**profit_and_loss_data)
            # Validate income statement
            issues = validator.validate_profit_and_loss(profit_and_loss)
            if issues:
                logger.warning("Profit and Loss validation issues:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
        except Exception as e:
            logger.warning(f"Error parsing profit and loss: {e}")

    # Step 3: Create final result
    result = FinancialStatement(
        company_info=company_info,
        balance_sheet=balance_sheet,
        profit_and_loss=profit_and_loss,
        metadata={
            "source_file": str(pdf_path),
            "total_pages": len(pages_data),
            "balance_sheet_pages": identified_pages.get("balance_sheet_pages", []),
            "profit_and_loss_pages": identified_pages.get("profit_and_loss_pages", []),
            "extraction_reasoning": reasoning,
        },
    )

    # Step 4: Save output
    if output_path:
        output_file = Path(output_path)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result.dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Output saved to: {output_file}")

    logger.info("Processing completed successfully!")
    return result


# ============================================================================
# STEP 7: CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Process Thai Financial Statements (DBD format) into structured JSON"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path (default: same as input with .json extension)",
    )
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env variable)",
        default=os.getenv("ANTHROPIC_API_KEY"),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    logger.info("Starting financial statement processor...")

    # Check API key
    if not args.api_key:
        print("ERROR: Anthropic API key not provided.")
        print("Set ANTHROPIC_API_KEY environment variable or use --api-key argument")
        exit(1)

    # Set default output path
    output_path = args.output
    if not output_path:
        pdf_path = Path(args.pdf_path)
        output_path = pdf_path.with_suffix(".json")

    try:
        # Process the financial statement
        result = process_financial_statement(
            pdf_path=args.pdf_path,
            anthropic_api_key=args.api_key,
            output_path=output_path,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)

        if result.company_info.name:
            print(f"Company: {result.company_info.name}")
        if result.company_info.period_end:
            print(f"Period: {result.company_info.period_end}")

        if result.balance_sheet:
            print(f"\n✓ Balance Sheet extracted")
            if result.balance_sheet.assets.total_assets:
                print(
                    f"  Total Assets: {result.balance_sheet.assets.total_assets:,.2f}"
                )
        else:
            print(f"\n✗ Balance Sheet not found")

        if result.profit_and_loss:
            print(f"\n✓ Income Statement extracted")
            if result.profit_and_loss.revenue:
                print(f"  Revenue: {result.profit_and_loss.revenue:,.2f}")
            if result.profit_and_loss.net_profit:
                print(f"  Net Profit: {result.profit_and_loss.net_profit:,.2f}")
        else:
            print(f"\n✗ Income Statement not found")

        print(f"\n✓ Output saved to: {output_path}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        exit(1)
