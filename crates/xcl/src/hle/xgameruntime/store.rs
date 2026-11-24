//! `XCL_STORE=offline-error`: XStore as on a console whose store service cannot be reached.
//! A store context opens, and every query that needs the service – the game's licence, add-on
//! licences, products, licence acquisition, consumables, user ids, updates, store UI – fails with
//! the GDK's E_GAMESTORE_NETWORK_ERROR. Nothing grants a licence: a licence handle never exists,
//! and `XStoreIsLicenseValid` (an unimplemented bool slot) answers false. Off by default: the
//! class is then refused at QueryApiImpl (`api::REFUSED`).

use std::ffi::c_void;
use std::sync::OnceLock;

use super::xasync::*;
use crate::util::{E_INVALIDARG, HRESULT, S_OK};

/// "The game failed to communicate with the store network." (XGameErr.h)
pub const E_GAMESTORE_NETWORK_ERROR: HRESULT = 0x8924_5301u32 as i32;

pub fn offline_error() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_STORE").is_ok_and(|v| v.eq_ignore_ascii_case("offline-error")))
}

static CONTEXT: u8 = 0;
static CALL: u8 = 0;

pub unsafe extern "system" fn XStoreCreateContext(_user: *mut c_void, out: *mut *const c_void) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    *out = (&CONTEXT as *const u8).cast();
    S_OK
}

pub unsafe extern "system" fn XStoreCloseContextHandle(_h: *mut c_void) {}

unsafe fn unreachable_store(b: *mut XAsyncBlock) -> HRESULT {
    if b.is_null() {
        return E_INVALIDARG;
    }
    begin_with(b, &CALL as *const u8 as usize, || Err(E_GAMESTORE_NETWORK_ERROR))
}

/// Every result function: the call's status, which is the network error.
pub unsafe extern "system" fn result(b: *mut XAsyncBlock) -> HRESULT {
    XAsyncGetStatus(b, false)
}

/// Licence-change and licence-lost registrations: accepted; offline nothing ever changes.
pub unsafe extern "system" fn register(_a: usize, _queue: usize, _ctx: usize, _cb: usize, token: *mut u64) -> HRESULT {
    if !token.is_null() {
        *token = 1;
    }
    S_OK
}

macro_rules! store_async {
    ($($name:ident($($arg:ident),*);)*) => {
        $(
            pub unsafe extern "system" fn $name($($arg: usize,)* b: *mut XAsyncBlock) -> HRESULT {
                $(let _ = $arg;)*
                unreachable_store(b)
            }
        )*
        /// (GDK name, function) of every async query, for the class's slots.
        pub fn async_queries() -> Vec<(&'static str, usize)> {
            vec![$((stringify!($name), $name as usize)),*]
        }
    };
}

store_async! {
    XStoreQueryAssociatedProductsAsync(ctx, kinds, per_page);
    XStoreQueryAssociatedProductsForStoreIdAsync(ctx, store_id, kinds, per_page);
    XStoreQueryProductsAsync(ctx, kinds, ids, id_count, actions, action_count);
    XStoreQueryEntitledProductsAsync(ctx, kinds, per_page);
    XStoreQueryProductForCurrentGameAsync(ctx);
    XStoreQueryProductForPackageAsync(ctx, kinds, package);
    XStoreProductsQueryNextPageAsync(query);
    XStoreAcquireLicenseForPackageAsync(ctx, package);
    XStoreAcquireLicenseForDurablesAsync(ctx, store_id);
    XStoreCanAcquireLicenseForStoreIdAsync(ctx, store_id);
    XStoreCanAcquireLicenseForPackageAsync(ctx, package);
    XStoreQueryGameLicenseAsync(ctx);
    XStoreQueryAddOnLicensesAsync(ctx);
    XStoreQueryConsumableBalanceRemainingAsync(ctx, store_id);
    XStoreReportConsumableFulfillmentAsync(ctx, store_id, quantity, tracking_id);
    XStoreGetUserCollectionsIdAsync(ctx, ticket, publisher_user_id);
    XStoreGetUserPurchaseIdAsync(ctx, ticket, publisher_user_id);
    XStoreQueryLicenseTokenAsync(ctx, ids, count, developer_string);
    XStoreShowPurchaseUIAsync(ctx, store_id, name, json);
    XStoreShowRateAndReviewUIAsync(ctx);
    XStoreShowRedeemTokenUIAsync(ctx, token, ids, count, no_csv);
    XStoreShowAssociatedProductsUIAsync(ctx, store_id, kinds);
    XStoreShowProductPageUIAsync(ctx, store_id);
    XStoreShowGiftingUIAsync(ctx, store_id, name, json);
    XStoreQueryGameAndDlcPackageUpdatesAsync(ctx);
    XStoreQueryPackageUpdatesAsync(ctx, packages, count);
    XStoreDownloadPackageUpdatesAsync(ctx, packages, count);
    XStoreDownloadAndInstallPackageUpdatesAsync(ctx, packages, count);
    XStoreDownloadAndInstallPackagesAsync(ctx, store_ids, count);
}

/// The class's functions: the context, every async query, their result functions and the
/// change registrations. The rest (enumeration of a query that never exists, licence handles)
/// stays unimplemented: E_NOTIMPL, or false for the bool slots.
pub fn functions() -> Vec<(&'static str, usize)> {
    let mut v = vec![
        ("XStoreCreateContext", XStoreCreateContext as usize),
        ("XStoreCloseContextHandle", XStoreCloseContextHandle as usize),
        ("XStoreRegisterGameLicenseChanged", register as usize),
        ("XStoreRegisterPackageLicenseLost", register as usize),
    ];
    for (name, f) in async_queries() {
        v.push((name, f));
    }
    for &(name, ..) in super::gdk_table::API.iter() {
        if name.starts_with("XStore") && (name.ends_with("Result") || name.ends_with("ResultCount") || name.ends_with("ResultSize")) {
            v.push((name, result as usize));
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_async_query_of_the_gdk_table_fails_with_the_network_error() {
        let ours: Vec<&str> = async_queries().iter().map(|q| q.0).collect();
        for &(name, ..) in super::super::gdk_table::API.iter() {
            if name.starts_with("XStore") && name.ends_with("Async") {
                assert!(ours.contains(&name), "{name} has no offline-error answer");
            }
        }
    }
}
