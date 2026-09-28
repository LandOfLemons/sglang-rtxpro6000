"""Focused daily-email report tests; no network and no mail delivery."""
import contextlib
import importlib.util
import datetime as dt
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

scripts = Path(__file__).parent
spec = importlib.util.spec_from_file_location('report', scripts / 'traffic_daily_report.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class FakeGitHub:
    repository = 'jpezzulli/sglang-rtxpro6000'
    def __init__(self):
        self.blobs = {
            'daily': {'repository': self.repository, 'schema_version': 1,
                      'collected_at': '2026-09-21T01:00:00Z',
                      'launch_date': '2026-09-20',
                      'coverage': {'complete_through_latest_exposed': True,
                                   'missing_dates_through_latest_exposed': [],
                                   'through_date': '2026-09-20'},
                      'days': {'2026-09-20': {'views': {'count': 10, 'uniques': 4},
                                              'clones': {'count': 8, 'uniques': 3}}}},
            'package': {'repository': self.repository, 'schema_version': 1,
                        'collected_at': '2026-09-21T01:00:00Z',
                        'days': {'2026-09-19': {'total_downloads': 40},
                                 '2026-09-20': {'total_downloads': 42}}},
            'metrics': {'repository': self.repository, 'schema_version': 1,
                        'collected_at': '2026-09-21T01:00:00Z',
                        'days': {'2026-09-19': {'stars': 8, 'forks': 2},
                                 '2026-09-20': {'stars': 9, 'forks': 2}}},
            'prior_snapshot': {'traffic': {
                'referrers': [{'referrer': 'reddit.com', 'count': 3, 'uniques': 2}],
                'paths': [{'path': '/jpezzulli/sglang-rtxpro6000', 'count': 4, 'uniques': 3}],
            }},
        }
    def api(self, path):
        if path.startswith('git/ref/heads/'):
            return {'object': {'sha': 'head'}}
        if path == 'git/trees/head':
            return {'truncated': False, 'tree': [
                {'path': 'daily.json', 'type': 'blob', 'sha': 'daily'},
                {'path': 'package-downloads.json', 'type': 'blob', 'sha': 'package'},
                {'path': 'repository-metrics.json', 'type': 'blob', 'sha': 'metrics'},
            ]}
        if path == 'git/trees/head?recursive=1':
            return {'truncated': False, 'tree': [
                {'path': 'snapshots/2026-09-20/prior.json', 'type': 'blob', 'sha': 'prior_snapshot'},
                {'path': 'snapshots/2026-09-21/latest.json', 'type': 'blob', 'sha': 'latest_snapshot'},
            ]}
        if path.startswith('git/blobs/'):
            import base64, json
            return {'encoding': 'base64', 'content': base64.b64encode(
                json.dumps(self.blobs[path.rsplit('/', 1)[1]]).encode()).decode()}
        if path == 'traffic/views?per=day':
            return {'count': 15, 'uniques': 6, 'views': [
                {'timestamp': '2026-09-20T00:00:00Z', 'count': 10, 'uniques': 4},
                {'timestamp': '2026-09-21T00:00:00Z', 'count': 5, 'uniques': 3}]}
        if path == 'traffic/clones?per=day':
            return {'count': 10, 'uniques': 4, 'clones': [
                {'timestamp': '2026-09-20T00:00:00Z', 'count': 8, 'uniques': 3},
                {'timestamp': '2026-09-21T00:00:00Z', 'count': 2, 'uniques': 2}]}
        if path == 'traffic/popular/referrers?per=day':
            return [{'referrer': 'reddit.com', 'count': 5, 'uniques': 4}]
        if path == 'traffic/popular/paths?per=day':
            return [{'path': '/jpezzulli/sglang-rtxpro6000', 'count': 6, 'uniques': 5}]
        raise AssertionError(path)


class StaleTrafficGitHub(FakeGitHub):
    """2026-09-28 collection whose returned traffic days stop at 2026-09-23."""
    def __init__(self):
        super().__init__()
        for name in ('daily', 'package', 'metrics'):
            self.blobs[name]['collected_at'] = '2026-09-28T01:00:00Z'
        daily = self.blobs['daily']
        daily['coverage']['through_date'] = '2026-09-23'
        daily['days'] = {day: {'views': {'count': 10, 'uniques': 4},
                               'clones': {'count': 8, 'uniques': 3}}
                         for day in r.dates('2026-09-20', '2026-09-23')}
        daily['days']['2026-09-23'] = {'views': {'count': 275, 'uniques': 120},
                                       'clones': {'count': 30, 'uniques': 12}}
        self.blobs['package']['days'] = {'2026-09-22': {'total_downloads': 42},
                                        '2026-09-23': {'total_downloads': 45}}
        self.blobs['metrics']['days'] = {'2026-09-22': {'stars': 9, 'forks': 2},
                                         '2026-09-23': {'stars': 11, 'forks': 3}}
    def api(self, path):
        if path == 'traffic/views?per=day':
            return {'count': 275, 'uniques': 120, 'views': [
                {'timestamp': '2026-09-23T00:00:00Z', 'count': 275, 'uniques': 120}]}
        if path == 'traffic/clones?per=day':
            return {'count': 30, 'uniques': 12, 'clones': [
                {'timestamp': '2026-09-23T00:00:00Z', 'count': 30, 'uniques': 12}]}
        return super().api(path)


class LiveRegressionGitHub(FakeGitHub):
    """2026-09-28 collection whose archive reaches 2026-09-27 but the API returns only 2026-09-23."""
    def __init__(self):
        super().__init__()
        for name in ('daily', 'package', 'metrics'):
            self.blobs[name]['collected_at'] = '2026-09-28T01:00:00Z'
        self.blobs['daily']['coverage']['through_date'] = '2026-09-27'
        self.blobs['daily']['days'] = {
            day: {'views': {'count': 10, 'uniques': 4}, 'clones': {'count': 8, 'uniques': 3}}
            for day in r.dates('2026-09-20', '2026-09-27')}
        self.blobs['package']['days'] = {'2026-09-27': {'total_downloads': 42}}
        self.blobs['metrics']['days'] = {'2026-09-27': {'stars': 9, 'forks': 2}}
    def api(self, path):
        if path == 'traffic/views?per=day':
            return {'count': 30, 'uniques': 12, 'views': [
                {'timestamp': f'2026-09-{day}T00:00:00Z', 'count': 10, 'uniques': 4}
                for day in ('21', '22', '23')]}
        if path == 'traffic/clones?per=day':
            return {'count': 24, 'uniques': 9, 'clones': [
                {'timestamp': f'2026-09-{day}T00:00:00Z', 'count': 8, 'uniques': 3}
                for day in ('21', '22', '23')]}
        return super().api(path)


class PartialArchiveGitHub(FakeGitHub):
    """2026-09-28 collection: live response ends 2026-09-23, contiguous archive ends 2026-09-25."""
    def __init__(self):
        super().__init__()
        for name in ('daily', 'package', 'metrics'):
            self.blobs[name]['collected_at'] = '2026-09-28T01:00:00Z'
        self.blobs['daily']['coverage']['through_date'] = '2026-09-25'
        self.blobs['daily']['days'] = {
            day: {'views': {'count': 10, 'uniques': 4}, 'clones': {'count': 8, 'uniques': 3}}
            for day in r.dates('2026-09-20', '2026-09-25')}
        self.blobs['package']['days'] = {'2026-09-25': {'total_downloads': 42}}
        self.blobs['metrics']['days'] = {'2026-09-25': {'stars': 9, 'forks': 2}}
    def api(self, path):
        if path == 'traffic/views?per=day':
            return {'count': 10, 'uniques': 4, 'views': [
                {'timestamp': '2026-09-23T00:00:00Z', 'count': 10, 'uniques': 4}]}
        if path == 'traffic/clones?per=day':
            return {'count': 8, 'uniques': 3, 'clones': [
                {'timestamp': '2026-09-23T00:00:00Z', 'count': 8, 'uniques': 3}]}
        return super().api(path)


class ArchiveThroughTodayGitHub(FakeGitHub):
    """2026-09-28 collection: the archive records today, but the live response stops at 2026-09-23."""
    def __init__(self):
        super().__init__()
        for name in ('daily', 'package', 'metrics'):
            self.blobs[name]['collected_at'] = '2026-09-28T01:00:00Z'
        self.blobs['daily']['coverage']['through_date'] = '2026-09-28'
        self.blobs['daily']['days'] = {
            day: {'views': {'count': 10, 'uniques': 4}, 'clones': {'count': 8, 'uniques': 3}}
            for day in r.dates('2026-09-20', '2026-09-28')}
        self.blobs['package']['days'] = {'2026-09-28': {'total_downloads': 42}}
        self.blobs['metrics']['days'] = {'2026-09-28': {'stars': 9, 'forks': 2}}
    def api(self, path):
        if path == 'traffic/views?per=day':
            return {'count': 10, 'uniques': 4, 'views': [
                {'timestamp': '2026-09-23T00:00:00Z', 'count': 10, 'uniques': 4}]}
        if path == 'traffic/clones?per=day':
            return {'count': 8, 'uniques': 3, 'clones': [
                {'timestamp': '2026-09-23T00:00:00Z', 'count': 8, 'uniques': 3}]}
        return super().api(path)


class DailyReportTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch.object(r, 'now', return_value=dt.datetime(
            2026, 9, 21, 10, tzinfo=dt.timezone.utc))
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def test_report_merges_live_window_and_labels_uniques_honestly(self):
        body, values, status = r.report(FakeGitHub(), 'traffic-history')
        self.assertEqual(values, {'stars': 9, 'forks': 2, 'views': 15, 'clones': 10,
                                  'daily_unique_cloners': 5, 'package_downloads': 42})
        self.assertIn('Stars: 9 (+1)', body)
        self.assertIn('Clones: 10', body)
        self.assertIn('sum_of_daily_unique_cloners: 5', body)
        self.assertIn('does not expose a deduplicated lifetime cloner count', body)
        self.assertIn('reddit.com: 5 views / 4 unique visitors (+2 views, +2 uniques)', body)
        self.assertFalse(status['stale'])
        self.assertEqual(status['live_day'], '2026-09-21')

    def test_first_report_uses_archived_daily_delta(self):
        body, _, _ = r.report(FakeGitHub(), 'traffic-history')
        self.assertIn('Stars: 9 (+1)', body)
        self.assertIn('Container downloads: 42 (+2)', body)
        self.assertIn('Views: 15', body)
        self.assertIn('Clones: 10', body)

    def test_cumulative_traffic_has_no_synthetic_latest_day_delta(self):
        # Repeating the last returned day's counts as a '(+N on <day>)' increase is
        # what made stale traffic look like fresh growth. Daily activity carries counts.
        body, _, _ = r.report(FakeGitHub(), 'traffic-history')
        for line in body.splitlines():
            self.assertNotIn(' on 2026-09-21)', line)
        self.assertNotIn('(+5', body)
        self.assertNotIn('(+2 on', body)

    def test_stale_traffic_still_emails_but_warns_in_body_and_subject(self):
        clock = patch.object(r, 'now', return_value=dt.datetime(
            2026, 9, 28, 10, tzinfo=dt.timezone.utc))
        with clock:
            body, values, status = r.report(StaleTrafficGitHub(), 'traffic-history')
            subject = r.subject_line('2026-09-28', status)
        self.assertEqual(values, {'stars': 11, 'forks': 3, 'views': 305, 'clones': 54,
                                  'daily_unique_cloners': 21, 'package_downloads': 45})
        self.assertTrue(status['stale'])
        self.assertEqual(status['live_day'], '2026-09-23')
        self.assertEqual(status['recorded_through'], '2026-09-23')
        self.assertEqual(status['missing'], ['2026-09-24', '2026-09-25',
                                             '2026-09-26', '2026-09-27'])
        self.assertEqual(status['unrecorded'], status['missing'])
        self.assertIn('WARNING', body)
        self.assertIn('returned no day after 2026-09-23 UTC', body)
        self.assertIn('4 day(s) are absent from the current API response '
                      '(2026-09-24 through 2026-09-27 UTC); none of them is recorded in '
                      'the archive.', body)
        self.assertIn('Data through: 2026-09-23 UTC', body)
        self.assertIn('Newest day in the current API response: 2026-09-23 UTC.', body)
        self.assertIn('recorded through 2026-09-23 UTC', body)
        self.assertIn('Traffic totals include only the recorded dates above; the current '
                      'API response is stale.', body)
        self.assertIn('2026-09-23 activity (last recorded day, 5 day(s) old; historical)', body)
        self.assertIn('Views: 275 from 120 daily unique visitors', body)
        self.assertIn('Views: 305', body)
        self.assertNotIn('(+275', body)
        self.assertNotIn('(+30 on', body)
        self.assertIn('the referrer/path response itself reports no source timestamp', body)
        self.assertIn('+0 delta does not prove there was no new traffic', body)
        self.assertIn('WARNING stale traffic, newest returned day 2026-09-23 '
                      '(recorded through 2026-09-23) —', subject)

    def test_live_window_regression_behind_archive_is_not_called_an_archive_gap(self):
        clock = patch.object(r, 'now', return_value=dt.datetime(
            2026, 9, 28, 10, tzinfo=dt.timezone.utc))
        with clock:
            body, values, status = r.report(LiveRegressionGitHub(), 'traffic-history')
            subject = r.subject_line('2026-09-28', status)
        self.assertEqual(values['views'], 80)
        self.assertEqual(values['daily_unique_cloners'], 24)
        self.assertTrue(status['stale'])
        self.assertEqual(status['live_day'], '2026-09-23')
        self.assertEqual(status['recorded_through'], '2026-09-27')
        self.assertEqual(status['missing'], ['2026-09-24', '2026-09-25',
                                             '2026-09-26', '2026-09-27'])
        self.assertEqual(status['unrecorded'], [])
        self.assertIn('4 day(s) are absent from the current API response '
                      '(2026-09-24 through 2026-09-27 UTC); the archive already records '
                      '2026-09-24 through 2026-09-27 UTC.', body)
        self.assertNotIn('not recorded in the archive', body)
        self.assertIn('Data through: 2026-09-27 UTC.', body)
        self.assertIn('Newest day in the current API response: 2026-09-23 UTC.', body)
        self.assertIn('Traffic since launch (cumulative over all recorded days, '
                      'recorded through 2026-09-27 UTC)', body)
        self.assertIn('2026-09-27 activity (latest completed UTC day', body)
        self.assertIn('WARNING stale traffic, newest returned day 2026-09-23 '
                      '(recorded through 2026-09-27) —', subject)

    def test_partial_archive_coverage_names_both_spans(self):
        clock = patch.object(r, 'now', return_value=dt.datetime(
            2026, 9, 28, 10, tzinfo=dt.timezone.utc))
        with clock:
            body, values, status = r.report(PartialArchiveGitHub(), 'traffic-history')
            subject = r.subject_line('2026-09-28', status)
        self.assertEqual(values, {'stars': 9, 'forks': 2, 'views': 60, 'clones': 48,
                                  'daily_unique_cloners': 18, 'package_downloads': 42})
        self.assertTrue(status['stale'])
        self.assertEqual(status['live_day'], '2026-09-23')
        self.assertEqual(status['recorded_through'], '2026-09-25')
        self.assertEqual(status['missing'], ['2026-09-24', '2026-09-25',
                                            '2026-09-26', '2026-09-27'])
        self.assertEqual(status['unrecorded'], ['2026-09-26', '2026-09-27'])
        self.assertIn('4 day(s) are absent from the current API response '
                      '(2026-09-24 through 2026-09-27 UTC); the archive records '
                      '2026-09-24 through 2026-09-25 UTC but not 2026-09-26 through '
                      '2026-09-27 UTC.', body)
        self.assertNotIn('none of them is recorded in the archive', body)
        self.assertNotIn('already records', body)
        self.assertIn('Data through: 2026-09-25 UTC.', body)
        self.assertIn('2026-09-25 activity (last recorded day, 3 day(s) old; historical)', body)
        self.assertIn('(recorded through 2026-09-25) —', subject)

    def test_stale_live_window_with_archive_through_today_keeps_partial_today_label(self):
        clock = patch.object(r, 'now', return_value=dt.datetime(
            2026, 9, 28, 10, tzinfo=dt.timezone.utc))
        with clock:
            body, values, status = r.report(ArchiveThroughTodayGitHub(), 'traffic-history')
            subject = r.subject_line('2026-09-28', status)
        self.assertEqual(values, {'stars': 9, 'forks': 2, 'views': 90, 'clones': 72,
                                  'daily_unique_cloners': 27, 'package_downloads': 42})
        self.assertTrue(status['stale'])
        self.assertEqual(status['live_day'], '2026-09-23')
        self.assertEqual(status['recorded_through'], '2026-09-28')
        self.assertEqual(status['age_days'], 0)
        self.assertEqual(status['missing'], ['2026-09-24', '2026-09-25',
                                            '2026-09-26', '2026-09-27'])
        self.assertEqual(status['unrecorded'], [])
        self.assertIn('WARNING: the current GitHub traffic response returned no day after '
                      '2026-09-23 UTC,', body)
        self.assertIn('4 day(s) are absent from the current API response '
                      '(2026-09-24 through 2026-09-27 UTC); the archive already records '
                      '2026-09-24 through 2026-09-27 UTC.', body)
        self.assertIn('Data through: 2026-09-28 UTC (today\'s bucket is partial', body)
        self.assertIn('2026-09-28 activity (partial UTC day so far', body)
        self.assertIn('Traffic totals include only the recorded dates above; the current '
                      'API response is stale.', body)
        self.assertNotIn('are not today\'s traffic', body)
        self.assertIn('WARNING stale traffic, newest returned day 2026-09-23 '
                      '(recorded through 2026-09-28) —', subject)

    def test_yesterday_data_is_fresh_without_warning(self):
        github = FakeGitHub()
        for name in ('daily', 'package', 'metrics'):
            github.blobs[name]['collected_at'] = '2026-09-22T01:00:00Z'
        clock = patch.object(r, 'now', return_value=dt.datetime(
            2026, 9, 22, 10, tzinfo=dt.timezone.utc))
        with clock:
            body, _, status = r.report(github, 'traffic-history')
            subject = r.subject_line('2026-09-22', status)
        self.assertFalse(status['stale'])
        self.assertEqual(status['live_day'], '2026-09-21')
        self.assertEqual(status['recorded_through'], '2026-09-21')
        self.assertEqual(status['missing'], [])
        self.assertEqual(status['age_days'], 1)
        self.assertNotIn('WARNING', body)
        self.assertNotIn('partial', body)
        self.assertIn('Data through: 2026-09-21 UTC.', body)
        self.assertIn('2026-09-21 activity (latest completed UTC day; GitHub can still '
                      'revise counts)', body)
        self.assertNotIn('complete)', body)
        self.assertNotIn('WARNING', subject)

    def test_current_utc_day_is_labelled_partial_not_stale(self):
        body, _, status = r.report(FakeGitHub(), 'traffic-history')
        self.assertEqual(status['live_day'], '2026-09-21')
        self.assertEqual(status['recorded_through'], '2026-09-21')
        self.assertEqual(status['age_days'], 0)
        self.assertFalse(status['stale'])
        self.assertNotIn('WARNING', body)
        self.assertIn('Data through: 2026-09-21 UTC (today\'s bucket is partial', body)
        self.assertIn('2026-09-21 activity (partial UTC day so far', body)

    def test_stale_boundary_starts_one_day_before_yesterday(self):
        github = FakeGitHub()
        for name in ('daily', 'package', 'metrics'):
            github.blobs[name]['collected_at'] = '2026-09-23T01:00:00Z'
        clock = patch.object(r, 'now', return_value=dt.datetime(
            2026, 9, 23, 10, tzinfo=dt.timezone.utc))
        with clock:
            body, _, status = r.report(github, 'traffic-history')
        self.assertTrue(status['stale'])
        self.assertEqual(status['live_day'], '2026-09-21')
        self.assertEqual(status['missing'], ['2026-09-22'])
        self.assertEqual(status['unrecorded'], ['2026-09-22'])
        self.assertIn('WARNING', body)
        self.assertIn('1 day(s) are absent from the current API response '
                      '(2026-09-22 UTC); none of them is recorded in the archive.', body)

    def test_future_dated_traffic_fails_without_report(self):
        rolling = {'views': {'days': {'2026-09-22': {}}}, 'clones': {'days': {}}}
        with self.assertRaises(RuntimeError):
            r.traffic_status(dt.date(2026, 9, 21), rolling, {'2026-09-20': {}})
        with self.assertRaises(RuntimeError):
            r.traffic_status(dt.date(2026, 9, 21), rolling, {'2026-10-01': {}})

    def test_main_emails_stale_subject_and_body_without_sending(self):
        run = patch.object(r.subprocess, 'run', return_value=Mock(returncode=0))
        argv = patch.object(sys, 'argv', ['traffic_daily_report.py', '--repository',
                                          FakeGitHub.repository, '--recipient', 'j@example.com',
                                          '--mail-command', '/usr/bin/mail'])
        clock = patch.object(r, 'now', return_value=dt.datetime(
            2026, 9, 28, 10, tzinfo=dt.timezone.utc))
        with clock, patch.object(r, 'GitHub', return_value=StaleTrafficGitHub()), \
                run as sent, argv, contextlib.redirect_stdout(io.StringIO()):
            r.main()
        args, kwargs = sent.call_args
        self.assertEqual(['/usr/bin/mail', '-s', args[0][2], 'j@example.com'], args[0])
        self.assertIn('WARNING stale traffic, newest returned day 2026-09-23', args[0][2])
        self.assertIn('Pennyroyal daily GitHub report — 2026-09-28', args[0][2])
        self.assertIn('WARNING: the current GitHub traffic response returned no day after',
                      kwargs['input'])
        self.assertIn('Traffic since launch (cumulative over all recorded days, '
                      'recorded through 2026-09-23 UTC)', kwargs['input'])

    def test_main_dry_run_prints_subject_and_sends_nothing(self):
        run = patch.object(r.subprocess, 'run')
        argv = patch.object(sys, 'argv', ['traffic_daily_report.py', '--repository',
                                          FakeGitHub.repository, '--recipient', 'j@example.com',
                                          '--dry-run'])
        clock = patch.object(r, 'now', return_value=dt.datetime(
            2026, 9, 28, 10, tzinfo=dt.timezone.utc))
        with clock, patch.object(r, 'GitHub', return_value=StaleTrafficGitHub()), run as sent, \
                argv, contextlib.redirect_stdout(io.StringIO()) as out:
            r.main()
        text = out.getvalue()
        sent.assert_not_called()
        self.assertIn('Subject: WARNING stale traffic, newest returned day 2026-09-23', text)
        self.assertIn('WARNING: the current GitHub traffic response returned no day after', text)

    def test_stale_traffic_collection_failure_still_raises(self):
        github = StaleTrafficGitHub()
        github.blobs['daily']['collected_at'] = '2026-09-23T01:00:00Z'
        clock = patch.object(r, 'now', return_value=dt.datetime(
            2026, 9, 28, 10, tzinfo=dt.timezone.utc))
        with clock, self.assertRaises(RuntimeError):
            r.report(github, 'traffic-history')

    def test_incomplete_archive_fails_without_report(self):
        github = FakeGitHub()
        github.blobs['daily']['coverage']['complete_through_latest_exposed'] = False
        with self.assertRaises(RuntimeError):
            r.report(github, 'traffic-history')

    def test_stale_archive_fails_without_report(self):
        github = FakeGitHub()
        github.blobs['package']['collected_at'] = '2026-09-20T23:59:59Z'
        with self.assertRaises(RuntimeError):
            r.report(github, 'traffic-history')

    def test_pre_cutoff_capture_fails_without_report(self):
        github = FakeGitHub()
        github.blobs['daily']['collected_at'] = '2026-09-21T00:05:00Z'
        with self.assertRaises(RuntimeError):
            r.report(github, 'traffic-history')

    def test_missing_history_date_fails_without_report(self):
        github = FakeGitHub()
        github.blobs['daily']['coverage']['through_date'] = '2026-09-21'
        with self.assertRaises(RuntimeError):
            r.report(github, 'traffic-history')

    def test_gap_between_archive_and_live_window_fails_without_report(self):
        traffic = FakeGitHub().blobs['daily']
        rolling = {'views': {'total': {'count': 1, 'uniques': 1},
                             'days': {'2026-09-22': {'count': 1, 'uniques': 1}}},
                   'clones': {'total': {'count': 1, 'uniques': 1},
                              'days': {'2026-09-22': {'count': 1, 'uniques': 1}}}}
        with self.assertRaises(RuntimeError):
            r.current_days(traffic, rolling)

    def test_empty_live_overlap_cannot_erase_archived_traffic(self):
        traffic = FakeGitHub().blobs['daily']
        rolling = {'views': {'total': {'count': 0, 'uniques': 0},
                             'days': {'2026-09-20': {'count': 0, 'uniques': 0}}},
                   'clones': {'total': {'count': 8, 'uniques': 3},
                              'days': {'2026-09-20': {'count': 8, 'uniques': 3}}}}
        with self.assertRaises(RuntimeError):
            r.current_days(traffic, rolling)

    def test_malformed_previous_snapshot_fails_without_report(self):
        github = FakeGitHub()
        del github.blobs['prior_snapshot']['traffic']['paths']
        with self.assertRaises(RuntimeError):
            r.report(github, 'traffic-history')



if __name__ == '__main__':
    unittest.main()
