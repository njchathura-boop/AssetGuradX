from datetime import datetime, timedelta, timezone
from decimal import Decimal

from assetguardx_forecasting.labeling import belongs_to_split, label_window


T = datetime(2023, 1, 1, tzinfo=timezone.utc)
H = timedelta(hours=1)


def test_labels_confirmation_is_included_in_actual_purge_window():
    values = {T + i * H: Decimal(0) for i in range(8)}
    values[T + 6 * H] = values[T + 7 * H] = Decimal("-0.01")
    label = label_window(
        T, values, horizon_hours=6, threshold=Decimal("0.005"), active_at_anchor=False
    )
    assert label.onset is True and label.downside == Decimal("0.01")
    assert label.information_end == T + 7 * H
    assert not belongs_to_split(
        anchor=T, information_end=label.information_end, start=T, end=T + 7 * H
    )
    values[T + 7 * H] = None
    assert label_window(
        T, values, horizon_hours=6, threshold=Decimal("0.005"), active_at_anchor=False
    ).onset is None
