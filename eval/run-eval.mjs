const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";
const endpoint = `${backendUrl.replace(/\/$/, "")}/run`;

const cases = [
  {
    name: "traffic-drop",
    query: "Traffic dropped sharply this week. Diagnose the cause and give actions."
  },
  {
    name: "inventory-overstock",
    query: "Inventory is overstocked and conversion is falling. What should we do?"
  },
  {
    name: "roi-down",
    query: "Ad ROI has declined for two days. Find the problem and propose a plan."
  }
];

async function runCase(testCase) {
  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      query: testCase.query,
      context: {
        merchant_id: "demo-001"
      }
    })
  });

  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }

  const data = await res.json();
  const hasSuggestion = typeof data?.final_answer === "string" && data.final_answer.trim().length > 0;
  const stepCount = Array.isArray(data?.steps) ? data.steps.length : 0;

  console.log(
    `[${testCase.name}] suggestion=${hasSuggestion ? "yes" : "no"} steps=${stepCount} metrics=${data?.metrics ? "yes" : "no"}`
  );

  return hasSuggestion;
}

async function main() {
  let passed = 0;

  for (const testCase of cases) {
    try {
      const ok = await runCase(testCase);
      if (ok) {
        passed += 1;
      }
    } catch (error) {
      console.log(`[${testCase.name}] suggestion=no error=${error instanceof Error ? error.message : String(error)}`);
    }
  }

  console.log(`Summary: ${passed}/${cases.length} cases returned suggestions.`);

  if (passed !== cases.length) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
