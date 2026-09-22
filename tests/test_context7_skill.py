import os
import unittest
import json
import re

class TestContext7Skill(unittest.TestCase):
    def setUp(self):
        self.global_config_dir = r"C:\Users\salau\.gemini\config"
        self.global_skill_dir = os.path.join(self.global_config_dir, "skills", "context7")
        self.workspace_skill_dir = os.path.join(r"d:\RAG", ".agents", "skills", "context7")
        self.mcp_config_path = os.path.join(self.global_config_dir, "mcp_config.json")

    def _parse_frontmatter(self, filepath):
        self.assertTrue(os.path.exists(filepath), f"File does not exist: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n", content, re.DOTALL)
        self.assertIsNotNone(match, f"No YAML frontmatter found in {filepath}")
        frontmatter_text = match.group(1)
        
        data = {}
        for line in frontmatter_text.splitlines():
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                data[key.strip()] = val.strip()
        return data, content

    def test_global_context7_skill_exists_and_valid(self):
        skill_file = os.path.join(self.global_skill_dir, "SKILL.md")
        fm, content = self._parse_frontmatter(skill_file)
        self.assertEqual(fm.get("name"), "context7")
        self.assertIn("description", fm)
        self.assertIn("ctx7", content)
        self.assertIn("npx ctx7@latest library", content)
        self.assertIn("npx ctx7@latest docs", content)

    def test_global_context7_references(self):
        ref_dir = os.path.join(self.global_skill_dir, "references")
        self.assertTrue(os.path.isdir(ref_dir), "Global references directory missing")
        expected_refs = ["docs.md", "setup.md", "skills.md"]
        for ref in expected_refs:
            path = os.path.join(ref_dir, ref)
            self.assertTrue(os.path.exists(path), f"Missing reference: {ref}")
            with open(path, "r", encoding="utf-8") as f:
                ref_content = f.read()
            self.assertGreater(len(ref_content), 100, f"Reference {ref} is unexpectedly short")

    def test_global_find_docs_skill(self):
        find_docs_file = os.path.join(self.global_config_dir, "skills", "find-docs", "SKILL.md")
        fm, content = self._parse_frontmatter(find_docs_file)
        self.assertEqual(fm.get("name"), "find-docs")
        self.assertIn("npx ctx7@latest library", content)

    def test_workspace_context7_skill(self):
        workspace_file = os.path.join(self.workspace_skill_dir, "SKILL.md")
        fm, content = self._parse_frontmatter(workspace_file)
        self.assertEqual(fm.get("name"), "context7")
        self.assertTrue(os.path.exists(os.path.join(self.workspace_skill_dir, "references", "docs.md")))
        self.assertTrue(os.path.exists(os.path.join(self.workspace_skill_dir, "references", "setup.md")))
        self.assertTrue(os.path.exists(os.path.join(self.workspace_skill_dir, "references", "skills.md")))

    def test_mcp_config_context7(self):
        self.assertTrue(os.path.exists(self.mcp_config_path), "mcp_config.json does not exist")
        with open(self.mcp_config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("mcpServers", data)
        self.assertIn("context7", data["mcpServers"])
        c7_config = data["mcpServers"]["context7"]
        self.assertEqual(c7_config.get("type"), "http")
        self.assertEqual(c7_config.get("url"), "https://mcp.context7.com/mcp")

if __name__ == "__main__":
    unittest.main()
