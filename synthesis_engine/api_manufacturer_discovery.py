import time
import os
from datetime import datetime
import requests
from urllib.parse import urlparse


class ApiManufacturerDiscoveryService:
    """
    Runs Groq-powered agents to discover new manufacturers and store them via ApiManufacturerService.
    """

    def __init__(self, manufacturer_service):
        self.manufacturer_service = manufacturer_service
    
        # Initialize Groq client
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            from groq import Groq as GroqClient
            self.groq_client = GroqClient(api_key=groq_api_key)
        else:
            self.groq_client = None

        # Initialize Supabase connection details (REST interface)
        self.supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        self.supabase_table = os.getenv("SUPABASE_MANUFACTURERS_TABLE", "API_manufacturers")
        if self.supabase_url and self.supabase_key:
            self.supabase_headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        else:
            self.supabase_headers = None
    
        self.trusted_domains = {
            "pharmacompass.com",
            "pharmaoffer.com",
            "orangebook.fda.gov",
            "fda.gov",
            "ema.europa.eu",
            "cdsco.gov.in",
            "who.int",
            "dcat.org",
            "scrip.pharmaintelligence.informa.com",
        }

    def discover(self, api_name: str, country: str):
        api_name = (api_name or "").strip()
        country = (country or "").strip()

        if not api_name or not country:
            return {
                "success": False,
                "error": "API name and country are required for discovery.",
            }

        existing_records = self._fetch_existing_records(api_name, country)
        skip_list = sorted(
            {
                (rec.get("manufacturer") or "").strip().lower()
                for rec in existing_records
                if rec.get("manufacturer")
            }
        )
        batches = [
            skip_list[i : i + 30] for i in range(0, len(skip_list), 30)
        ] or [[]]

        discovered_records = []
        for batch in batches:
            discovered_records.extend(
                self._discover_with_groq(api_name, country, batch)
            )
            if discovered_records:
                break  # stop once we find fresh data

        if discovered_records:
            insert_result = self._insert_records(discovered_records, "groq_discovery")
            newly_inserted = insert_result["rows"]
            inserted_count = insert_result["inserted"]
        else:
            newly_inserted = []
            inserted_count = 0

        refreshed = self._fetch_existing_records(api_name, country)

        return {
            "success": True,
            "existing_records": existing_records,
            "new_records": newly_inserted,
            "all_records": refreshed,
            "inserted_count": inserted_count,
        }

    def _discover_with_groq(self, api_name, country, skip_batch):
        if not self.groq_client:
            return []

        try:
            time.sleep(1)  # Small delay to avoid hammering Groq API
            groq_output = self._run_groq_extraction(
                api_name=api_name,
                country=country,
                skip_list=skip_batch,
            )
            if groq_output:
                return self._extract_manufacturers(
                    groq_output, api_name, country, skip_batch
                )
        except Exception:
            pass
        return []
    
    def _run_groq_extraction(self, api_name: str, country: str, skip_list: list) -> str:
        """Use Groq directly to research and extract manufacturers"""
        if not self.groq_client:
            return None
            
        skip_clause = ", ".join(skip_list[:10]) if skip_list else "None"  # Limit skip list size
        trusted_clause = ", ".join(sorted(self.trusted_domains))
        
        prompt = f"""
You are a pharmaceutical business intelligence expert. Identify legitimate API manufacturers for "{api_name}" located in "{country}".

Skip these known manufacturers: {skip_clause}

Requirements:
- Provide only manufacturers that produce the API (not formulations) and operate in {country}.
- Verify each manufacturer using information from trusted public sources ({trusted_clause}). If no trusted citation exists, exclude the manufacturer.
- Return results as a markdown table with columns:
  | manufacturers | country | usdmf | cep | source_name | source_url |
- Provide HTTPS URLs pointing directly to the evidence page. Prefer regulatory listings or manufacturer catalogs.
- usdmf/cep should be "Yes"/"No"/"Unknown".
- Do not include duplicate manufacturers or any entry from the skip list.
"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a pharmaceutical data extraction expert. Return only valid markdown tables."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=2000
            )
            return response.choices[0].message.content if response.choices else None
        except Exception:
            return None

    def _extract_manufacturers(self, markdown_output, api_name, country, skip_batch):
        manufacturers = []
        if not markdown_output:
            return manufacturers

        lines = markdown_output.splitlines()
        existing_lower = {name.lower() for name in skip_batch}
        for line in lines:
            if "|" in line and not line.lower().startswith("| manufacturers") and not line.startswith("|---"):
                raw_parts = [p.strip() for p in line.split("|")]
                if len(raw_parts) < 8:
                    continue
                parts = raw_parts[1:-1]  # drop leading/trailing blanks caused by table edges
                if len(parts) < 6:
                    continue
                manu_name = parts[0]
                manu_lower = manu_name.lower()
                if manu_lower in existing_lower:
                    continue
                country_val = parts[1]
                usdmf = "Yes" if parts[2].strip().lower() in ["yes", "t"] else "No"
                cep = "Yes" if parts[3].strip().lower() in ["yes", "t"] else "No"
                source_name = parts[4]
                source_url = parts[5]

                if not self._is_valid_source(source_url):
                    continue

                if country.lower() in country_val.lower():
                    manufacturers.append(
                        {
                            "api_name": api_name,
                            "manufacturer": manu_name,
                            "country": country_val,
                            "usdmf": usdmf,
                            "cep": cep,
                            "source_name": source_name,
                            "source_url": source_url,
                        }
                    )
        return manufacturers

    def _is_valid_source(self, url: str) -> bool:
        if not url:
            return False
        parsed = urlparse(url.strip())
        if parsed.scheme.lower() != "https":
            return False
        domain = parsed.netloc.lower()
        return any(domain == trusted or domain.endswith(f".{trusted}") for trusted in self.trusted_domains)

    # ---------- Supabase helpers ----------

    def _use_supabase(self):
        return bool(self.supabase_headers and self.supabase_url)

    def _fetch_existing_records(self, api_name, country):
        if self._use_supabase():
            try:
                params = {
                    "select": "api_name,manufacturer,country,usdmf,cep,source_name,source_url",
                    "api_name": f"ilike.*{api_name}*",
                    "country": f"ilike.*{country}*",
                }
                response = requests.get(
                    f"{self.supabase_url}/rest/v1/{self.supabase_table}",
                    headers=self.supabase_headers,
                    params=params,
                    timeout=20,
                )
                response.raise_for_status()
                return response.json()
            except Exception:
                pass
        return self.manufacturer_service.query(api_name, country)

    def _insert_records(self, records, source_label):
        if not records:
            return {"inserted": 0, "rows": []}

        if self._use_supabase():
            try:
                payload = []
                import_ts = datetime.utcnow().isoformat()
                for rec in records:
                    payload.append(
                        {
                            "api_name": rec.get("api_name", ""),
                            "manufacturer": rec.get("manufacturer", ""),
                            "country": rec.get("country", ""),
                            "usdmf": rec.get("usdmf", ""),
                            "cep": rec.get("cep", ""),
                            "source_name": rec.get("source_name", ""),
                            "source_url": rec.get("source_url", ""),
                            "source_file": source_label,
                            "imported_at": import_ts,
                        }
                    )
                response = requests.post(
                    f"{self.supabase_url}/rest/v1/{self.supabase_table}",
                    headers={**self.supabase_headers, "Prefer": "return=representation"},
                    json=payload,
                    timeout=20,
                )
                response.raise_for_status()
                data = response.json()
                return {"inserted": len(data), "rows": data}
            except Exception:
                pass

        return self.manufacturer_service.insert_records(records, source_label)

