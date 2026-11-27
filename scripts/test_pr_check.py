import unittest

from pr_check import check


HEAD = "1234567890abcdef"
BODY = "**Tested**\nSynthetic fixtures\n**Not tested**\nRetail titles\n**AI use**\nNone\n"


def changed(path, binary=False):
    return (path, "modified", binary, lambda: 10)


class TestOnlyApprovalTests(unittest.TestCase):
    def test_runtime_path_requires_smoke_until_approved(self):
        files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        problems = check(BODY, HEAD, files, lambda _: False)
        self.assertEqual(2, len(problems))
        self.assertTrue(all("Regression run" in problem for problem in problems))
        self.assertEqual([], check(BODY, HEAD, files, lambda _: False, test_only=True))

    def test_approval_keeps_description_and_asset_checks(self):
        files = [
            changed("crates/xcl/src/hle/d3d12x/vram.rs"),
            changed("assets/content.rpf"),
            changed("notes/attachment.txt", binary=True),
        ]
        body = "**Tested**\nSynthetic fixtures\n**AI use**\nQwen\n"
        problems = check(body, HEAD, files, lambda _: False, test_only=True)
        self.assertEqual(4, len(problems))
        for reason in ("Not tested", "AI use", "content.rpf", "attachment.txt"):
            self.assertTrue(any(reason in problem for problem in problems), problems)

    def test_test_paths_do_not_need_approval(self):
        files = [changed("crates/xcl/tests/integration.rs"), changed("crates/xcl/src/gpu/recomp/tests.rs")]
        self.assertEqual([], check(BODY, HEAD, files, lambda _: False))

    def test_repo_assets_allow_small_non_game_binaries(self):
        files = [
            changed("assets/logo/XSEmu.ico", binary=True),
            changed("launcher/resources/icon.png", binary=True),
        ]
        self.assertEqual([], check(BODY, HEAD, files, lambda _: False))

    def test_repo_assets_reject_oversized_or_game_data_binaries(self):
        oversized = ("assets/logo/huge.ico", "modified", True, lambda: (1 << 20) + 1)
        game_data = changed("assets/logo/content.rpf", binary=True)
        problems = check(BODY, HEAD, [oversized, game_data], lambda _: False)
        self.assertEqual(2, len(problems))
        self.assertTrue(any("huge.ico" in p for p in problems), problems)
        self.assertTrue(any("content.rpf" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
