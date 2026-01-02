# 🏥 Immunization Analytics Dashboard Guide

This guide provides a detailed explanation of every visualization available across the five dashboards. It is designed to help stakeholders understand "What does this chart show?" and "How does this help me make decisions?".

---

## 1. Vaccine Field Dashboard (Main Operations)
**Goal:** High-level operational oversight of vaccine coverage, uptake, and major barriers.

### Key Performance Indicators (KPIs)
*   **Total Facility Visits:** Volume of children interacting with the health system.
*   **Active Zero-Dose Burden:** The absolute number of children currently identified as zero-dose and not yet resolved (KPI Source: Full Zero-Dose Database).
*   **Penta1-Penta3 Dropout:** Percentage of children who started but did not complete the primary Penta series. Target < 10%.
*   **Priority Settlements:** Number of settlements flagged for intervention in the current filtered view.

### Visualizations
1.  **Volume: Total Doses Administered**
    *   **What it shows:** Total count of doses given for each antigen.
    *   **Decision Support:** Identifies which vaccines are moving (high volume) vs. those with low uptake. Helps distinct between supply issues or demand issues.

2.  **Distribution by Schedule Stage**
    *   **What it shows:** Breaks down doses by the age stage they are intended for (e.g., Birth, 6 Weeks, 9 Months).
    *   **Decision Support:** Reveals bottlenecks in the schedule. For example, if "Birth" volume is high but "9 Months" is low, retention is the problem.

3.  **Antigen Coverage (%)**
    *   **What it shows:** Estimated coverage percentage for each antigen based on facility data.
    *   **Decision Support:** The primary metric for performance. Low bars indicate specific antigens needing campaign focus.

4.  **Timeliness: Coverage by Age Group**
    *   **What it shows:** A heatmap showing *when* children are receiving vaccines.
    *   **Decision Support:** Highlights delayed vaccination relative to the ideal schedule.

5.  **Dropout Rates by Series**
    *   **What it shows:** Dropout rates for specific pairings (e.g., Penta1→Penta3, Rota1→Rota3).
    *   **Decision Support:** Pinpoints exactly where in the multi-dose schedule children are dropping out.

6.  **Barriers to Access (Zero-Dose Drivers)**
    *   **What it shows:** The self-reported reasons why a child is Zero-Dose (e.g., "Lack of Information", "Distance").
    *   **Decision Support:** Taylor intervention messaging. If "Distance" is #1, plan mobile teams. If "Information" is #1, plan community senstization.

7.  **Zero-Dose Resolution by LGA**
    *   **What it shows:** Stacked bar chart of Active vs. Resolved cases per LGA.
    *   **Decision Support:** Tracks the progress of remediation teams. LGAs with high "Active" bars need more resources.

---

## 2. Gender Equity Dashboard
**Goal:** Ensure equal access and outcomes for male and female children.

### Key Performance Indicators
*   **Facility Visits & Zero-Dose Cases:** Broken down by total count to show the scale of data analyzed.
*   **Male vs. Female Visits:** Raw counts to spot immediate imbalances.

### Visualizations
1.  **Routine Access (Facility Visits)**
    *   **What it shows:** Percentage split of facility visits by gender.
    *   **Decision Support:** Is the system reaching boys and girls equally?

2.  **Active Burden (Zero-Dose Share)**
    *   **What it shows:** Percentage split of the *unresolved* zero-dose caseload by gender.
    *   **Decision Support:** Does one gender make up a disproportionate share of the unvaccinated population?

3.  **Barriers by Age Group (Settlement Data)**
    *   **What it shows:** A heatmap of reasons for zero-dose status, broken down by age group.
    *   **Decision Support:** Checks if barriers change as children get older (e.g., newborns might face "Cost" while older children face "Refusal").

4.  **Age Profile by Settlement (All Settlements)**
    *   **What it shows:** Activity/Burden for every single settlement, stacked by age group.
    *   **Decision Support:** The ultimate micro-planning tool. It shows exactly which settlements have the most burden and whether that burden is mostly Newborns (recent problem) or Older Children (legacy problem).

---

## 3. Age Analytics Dashboard
**Goal:** Analyze age-specific vulnerability and timeliness of vaccination.

### Key Performance Indicators
*   **Newborns (<6w):** Count of recently born zero-dose children. **Critical Priority** for immediate enrollment.
*   **Overaged (>2y):** Count of older zero-dose children who require catch-up campaigns.

### Visualizations
1.  **Vaccination Timeliness**
    *   **What it shows:** Density heatmap of *when* vaccines are administered vs. the child's age group.
    *   **Decision Support:** Identifies "Late Starters". If Penta 1 is being given to children >1 year old, routine outreach is failing.

2.  **Estimated Missed Doses**
    *   **What it shows:** Estimates the specific vaccines a child has missed based on their current age.
    *   **Decision Support:** Helps forecast supply needs for catch-up activities (e.g., specific need for Measles doses).

3.  **Resolution Success by Age**
    *   **What it shows:** Percentage of cases resolved for each age group.
    *   **Decision Support:** Shows which age groups are hardest to reach. often older children are harder to resolve than newborns.

4.  **Penta 3 Delay Analysis**
    *   **What it shows:** Box plot of the delay (in weeks) for Penta 3 administration across LGAs.
    *   **Decision Support:** Identifies districts with systemic delays in completing the schedule.

5.  **Zero-Dose Age-Gender Pyramid**
    *   **What it shows:** Population pyramid of the Zero-Dose list.
    *   **Decision Support:** Visualizes the demographic shape of the burden. A wide base means the problem is growing (more newborns).

6.  **Chronic vs New Issues (Avg Age by Settlement)**
    *   **What it shows:** The average age of zero-dose children in top settlements.
    *   **Decision Support:** 
        *   **Low Avg Age:** New/Emerging problem (Recent stockout? New barrier?).
        *   **High Avg Age:** Chronic/Legacy problem (Long-term neglect).

---

## 4. Household & Settlement Dashboard
**Goal:** Micro-planning and last-mile intervention targeting.

### Key Performance Indicators
*   **Settlements Mapped:** Number of unique communities with data.
*   **Total Active Burden:** Verified count of unresolved cases.
*   **Avg Distance:** Average distance to health facility for these households.

### Visualizations
1.  **All Settlements (Total Identified)**
    *   **What it shows:** Total historical zero-dose cases identified per settlement.
    *   **Decision Support:** Shows historical hotspots.

2.  **All Settlements (Active Unresolved)**
    *   **What it shows:** Current "To-Do List". Settlements with the most children waiting for vaccination.
    *   **Decision Support:** These are the immediate targets for mobile teams.

3.  **Priority Matrix (Burden vs Distance)**
    *   **What it shows:** Scatter plot comparing `Active Burden` (Y-axis) vs. `Distance to Facility` (X-axis).
    *   **Decision Support:**
        *   **Top Right:** **High Priority**. High burden, far away. Needs Mobile Teams.
        *   **Top Left:** **High Priority**. High burden, close by. Needs Community Engagement (why aren't they coming?).

4.  **Dominant ZD Barriers**
    *   **What it shows:** The single most common barrier reported in each LGA or Ward.
    *   **Decision Support:** Strategic messaging. Don't spend money on "Education" if the barrier is "Cost/Transport".

---

## 5. Time-Series Dashboard
**Goal:** Track trends over time.

### Visualizations
1.  **Zero-Dose Enrollment Trends**
    *   **What it shows:** The number of new zero-dose children identified (enrolled) per month, stacked by their current status (Active vs. Resolved).
    *   **Decision Support:**
        *   **Rising Curve:** Identification efforts are improving (good) or the problem is worsening (bad).
        *   **Widening Red Area (Resolved):** The program is successfully treating the children it finds.
        *   **Widening Blue Area (Active):** Identification is outpacing resolution capacity (Warning sign).
